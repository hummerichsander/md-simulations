#!/usr/bin/env python3
"""Two-state basin populations, kinetics and convergence from a collective variable.

Answers the question a free-energy surface cannot: does a trajectory *reversibly*
sample both basins, and are the populations converged? Consumes the ``.npz`` written
by :mod:`~md_simulations.analysis.cvs` rather than a trajectory, so re-running the
kinetics costs nothing and the CV definition is fixed upstream where it belongs.

Two design choices are worth knowing about.

**Core sets, not a Markov state model.** For a two-state system with separated cores,
a hysteretic assignment plus a direct dwell-time count *is* the mean-first-passage
estimator, and it needs no discretisation, no implied-timescale plateau and no
Chapman-Kolmogorov test. On 300 ns of trp-cage the MSM implied timescales do not
converge at all, so an MSM here yields precise-looking numbers with no validity; it is
kept as a diagnostic that documents the non-convergence. The usual objection to core
sets — arbitrary cutoffs — is answered by :func:`cutoff_scan`.

**Events, not crossings.** A single unfolding excursion recrosses any boundary many
times: on trp-cage at 330 K a plain threshold gives 470 crossings for 17 excursions, a
28-fold overcount. Hysteresis removes most of that and :func:`merge_events` removes the
rest by requiring a new state to be held before it counts.

Segments are never spliced. Each entry of ``segment_lengths`` in the ``.npz`` is treated
as an independent trajectory, so no transition, dwell time or lagged pair straddles two
replicas.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from md_simulations.analysis.fes import free_energy_1d

logger = logging.getLogger("md_simulations.analysis.basins")

# kJ mol^-1 K^-1 — melting-curve enthalpies come out in kJ/mol.
GAS_CONSTANT = 0.0083144626

UNFOLDED, FOLDED, UNASSIGNED = 0, 1, -1


def state_series(cv: np.ndarray, q_unfolded: float, q_folded: float) -> np.ndarray:
    """Assign frames to basins with a hysteretic core-set rule.

    A frame at or above ``q_folded`` is folded, at or below ``q_unfolded`` is unfolded,
    and anything between inherits the last core visited. Frames before the first core
    visit cannot be assigned and are marked ``-1``; every consumer here drops them.

    :param cv: Collective variable, shape ``(n_frames,)``, larger meaning more folded.
    :param q_unfolded: Upper edge of the unfolded core.
    :param q_folded: Lower edge of the folded core.
    :return: Array of ``0`` (unfolded), ``1`` (folded) or ``-1`` (unassigned).
    :raises ValueError: If the cores overlap."""
    if q_unfolded >= q_folded:
        raise ValueError(
            f"Cores overlap: q_unfolded={q_unfolded:.3f} must be below "
            f"q_folded={q_folded:.3f}."
        )
    cv = np.asarray(cv, dtype=float)
    marks = np.full(cv.shape, UNASSIGNED, dtype=np.int8)
    marks[cv >= q_folded] = FOLDED
    marks[cv <= q_unfolded] = UNFOLDED

    # Forward-fill the core marks through the intermediate region.
    known = marks >= 0
    if not known.any():
        return marks
    idx = np.where(known, np.arange(len(marks)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = marks[idx]
    filled[: np.argmax(known)] = UNASSIGNED
    return filled


def merge_events(states: np.ndarray, min_dwell_frames: int) -> np.ndarray:
    """Collapse basin visits shorter than ``min_dwell_frames`` into their neighbours.

    Without this, one physical excursion that jitters across a core boundary counts as
    many transitions. A visit only becomes an event once the trajectory commits to it
    for ``min_dwell_frames``.

    Absorption is iterative and always takes the *shortest* remaining short visit,
    flipping it to its neighbours' state so the runs on either side join up. Sweeping
    forward and absorbing into the preceding state instead would destroy a long
    excursion that happens to be finely fragmented — every fragment is short, so the
    whole excursion would vanish rather than becoming one event.

    :param states: Output of :func:`state_series`.
    :param min_dwell_frames: Minimum frames a state must be held to count.
    :return: A new array with short visits merged away."""
    out = np.asarray(states).copy()
    if min_dwell_frames <= 1:
        return out
    idx = np.flatnonzero(out >= 0)
    if len(idx) == 0:
        return out

    s = out[idx]
    bounds = np.r_[0, np.flatnonzero(np.diff(s)) + 1, len(s)]
    runs = [[int(s[a]), int(a), int(b)] for a, b in zip(bounds[:-1], bounds[1:])]

    while len(runs) > 1:
        lengths = [b - a for _, a, b in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_dwell_frames:
            break
        neighbour = k - 1 if k > 0 else k + 1
        runs[k][0] = runs[neighbour][0]
        collapsed = [runs[0]]
        for run in runs[1:]:
            if run[0] == collapsed[-1][0]:
                collapsed[-1][2] = run[2]
            else:
                collapsed.append(run)
        runs = collapsed

    for state, a, b in runs:
        s[a:b] = state
    out[idx] = s
    return out


def count_transitions(states: np.ndarray) -> tuple[int, int]:
    """Count basin changes in an assigned series.

    :param states: Output of :func:`state_series` (optionally merged).
    :return: ``(n_unfolded_to_folded, n_folded_to_unfolded)``."""
    seq = np.asarray(states)
    seq = seq[seq >= 0]
    if len(seq) < 2:
        return 0, 0
    changes = seq[np.r_[True, np.diff(seq) != 0]]
    prev, nxt = changes[:-1], changes[1:]
    return (
        int(np.sum((prev == UNFOLDED) & (nxt == FOLDED))),
        int(np.sum((prev == FOLDED) & (nxt == UNFOLDED))),
    )


def dwell_times(states: np.ndarray, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
    """Complete dwell times per basin, in ns.

    The first and last visits are censored — they were already running when the segment
    began, or still running when it ended — so only interior visits are returned. That
    keeps the mean an unbiased first-passage estimate at the cost of two visits.

    :param states: Output of :func:`state_series` (optionally merged).
    :param dt_ps: Time between consecutive frames in ps.
    :return: ``(folded_dwells, unfolded_dwells)`` in ns."""
    seq = np.asarray(states)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return np.array([]), np.array([])
    bounds = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
    runs = [(seq[a], b - a) for a, b in zip(bounds[:-1], bounds[1:])]
    interior = runs[1:-1]
    scale = dt_ps / 1000.0
    folded = np.array([n * scale for st, n in interior if st == FOLDED])
    unfolded = np.array([n * scale for st, n in interior if st == UNFOLDED])
    return folded, unfolded


def basin_populations(states: np.ndarray) -> tuple[float, float]:
    """Fraction of assigned frames in each basin.

    :param states: Output of :func:`state_series` (optionally merged).
    :return: ``(p_folded, p_unfolded)``; both NaN if nothing is assigned."""
    seq = np.asarray(states)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return float("nan"), float("nan")
    p_folded = float(np.mean(seq == FOLDED))
    return p_folded, 1.0 - p_folded


def block_estimates(states: np.ndarray, n_blocks: int = 5) -> np.ndarray:
    """Folded fraction in each of ``n_blocks`` consecutive blocks.

    A spread far larger than the sampling error is the signature of a non-stationary
    trajectory — e.g. one that starts folded and only unfolds halfway through.

    :param states: Output of :func:`state_series` (optionally merged).
    :param n_blocks: Number of equal, non-overlapping blocks.
    :return: Array of shape ``(n_blocks,)``, NaN where a block has no assigned frame."""
    out = []
    for block in np.array_split(np.asarray(states), n_blocks):
        valid = block[block >= 0]
        out.append(float(np.mean(valid == FOLDED)) if len(valid) else float("nan"))
    return np.array(out)


def block_bootstrap(
    states: np.ndarray,
    block_frames: int,
    n_resamples: int = 2000,
    seed: int = 0,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval on the folded fraction.

    Blocks preserve the correlation the series carries within ``block_frames``, which an
    i.i.d. bootstrap would destroy — giving a confidence interval far too narrow.

    :param states: Output of :func:`state_series` (optionally merged).
    :param block_frames: Block length in frames; use roughly 2 correlation times.
    :param n_resamples: Number of bootstrap resamples.
    :param seed: RNG seed.
    :param ci: Central confidence level in percent.
    :return: ``(ci_low, ci_high)`` on the folded fraction, or ``(nan, nan)``."""
    seq = np.asarray(states)
    seq = seq[seq >= 0]
    n = len(seq)
    block = max(1, min(int(block_frames), n))
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block)
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block)
    samples = seq[(starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)]
    means = (samples == FOLDED).mean(axis=1)
    half = (100.0 - ci) / 2.0
    return float(np.percentile(means, half)), float(np.percentile(means, 100.0 - half))


def integrated_autocorr_time(
    x: np.ndarray, dt_ps: float, c: float = 6.0
) -> tuple[float, int]:
    """Integrated autocorrelation time by Sokal's automatic windowing, in ns.

    The window is the smallest ``M`` with ``M >= c * tau(M)``; summing the noisy tail
    beyond it adds variance, not signal.

    :param x: A real-valued series, shape ``(n_frames,)``.
    :param dt_ps: Time between consecutive frames in ps.
    :param c: Sokal window factor; 6 is the usual choice.
    :return: ``(tau_ns, window_frames)``. ``tau`` is 0 for a constant series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0, 0
    x = x - x.mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n].real
    if acf[0] <= 0:
        return 0.0, 0
    acf /= acf[0]

    taus = 2.0 * np.cumsum(acf) - 1.0
    window = n - 1
    below = np.flatnonzero(np.arange(n) >= c * taus)
    if len(below):
        window = int(below[0])
    return float(max(taus[window], 0.0)) * dt_ps / 1000.0, window


def locate_basins(
    cv: np.ndarray,
    bins: int = 90,
    sigma: float = 2.0,
    x_range: tuple[float, float] | None = (0.0, 1.0),
    min_barrier: float = 0.5,
) -> tuple[float, float, float]:
    """Locate the two basin minima and the barrier between them on the 1-D FES.

    Uses ``f_max=inf`` deliberately: at low temperature the unfolded well sits far above
    any plotting ceiling, and a ceilinged surface would hide it entirely.

    :param cv: Collective variable, shape ``(n_frames,)``.
    :param bins: Histogram bins.
    :param sigma: Gaussian smoothing width in bins.
    :param x_range: Histogram span; ``None`` uses the data extent.
    :param min_barrier: Minimum barrier height in kT above the shallower minimum for the
        surface to count as two-state.
    :return: ``(cv_unfolded_min, cv_barrier, cv_folded_min)``.
    :raises ValueError: If the surface is not two-state — which is the correct verdict
        for a trajectory that never leaves one basin, not an error to be worked around."""
    from scipy.signal import find_peaks

    centres, F = free_energy_1d(cv, bins, sigma, np.inf, x_range)
    # Pad so a minimum sitting in the first or last bin is still found.
    padded = np.r_[-np.inf, np.where(np.isfinite(F), -F, -np.inf), -np.inf]
    peaks = find_peaks(padded)[0] - 1
    if len(peaks) < 2:
        raise ValueError(
            f"The free-energy surface has {len(peaks)} minimum/minima, so it is not "
            "two-state. The trajectory does not sample two basins along this CV."
        )

    lo, hi = sorted(peaks[np.argsort(F[peaks])[:2]])
    barrier = lo + int(np.nanargmax(F[lo : hi + 1]))
    height = F[barrier] - max(F[lo], F[hi])
    if height < min_barrier:
        raise ValueError(
            f"Barrier is only {height:.2f} kT above the shallower minimum "
            f"(need {min_barrier}); the two minima are not separated states."
        )
    return float(centres[lo]), float(centres[barrier]), float(centres[hi])


def core_cutoffs(
    cv: np.ndarray,
    bins: int = 90,
    sigma: float = 2.0,
    x_range: tuple[float, float] | None = (0.0, 1.0),
    depth: float = 1.0,
) -> tuple[float, float]:
    """Core-set edges placed where the FES rises ``depth`` kT above each minimum.

    Deriving the cores from the surface rather than hard-coding them means the same
    command works across temperatures whose basins sit in different places.

    :param cv: Collective variable, shape ``(n_frames,)``.
    :param bins: Histogram bins.
    :param sigma: Gaussian smoothing width in bins.
    :param x_range: Histogram span; ``None`` uses the data extent.
    :param depth: Height in kT above each minimum at which the core ends.
    :return: ``(q_unfolded, q_folded)`` for :func:`state_series`.
    :raises ValueError: Propagated from :func:`locate_basins` when not two-state."""
    centres, F = free_energy_1d(cv, bins, sigma, np.inf, x_range)
    cv_lo, cv_barrier, cv_hi = locate_basins(cv, bins, sigma, x_range)
    lo = int(np.argmin(np.abs(centres - cv_lo)))
    hi = int(np.argmin(np.abs(centres - cv_hi)))

    # Walk from each minimum toward the barrier until the surface has risen `depth`.
    q_unfolded = centres[lo]
    for i in range(lo, hi + 1):
        if np.isfinite(F[i]) and F[i] - F[lo] >= depth:
            break
        q_unfolded = centres[i]
    q_folded = centres[hi]
    for i in range(hi, lo - 1, -1):
        if np.isfinite(F[i]) and F[i] - F[hi] >= depth:
            break
        q_folded = centres[i]

    if q_unfolded >= q_folded:
        mid = 0.5 * (q_unfolded + q_folded)
        span = 0.02 * (centres[-1] - centres[0])
        q_unfolded, q_folded = mid - span, mid + span
    return float(q_unfolded), float(q_folded)


def reverse_cumulative_population(
    states: np.ndarray, n_points: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Folded fraction computed from frame ``i`` onward, as a function of ``i``.

    A stronger stationarity test than comparing FES halves, whose well *depths* agree
    long before the populations do. A trajectory that needed a burn-in shows a curve
    that drifts and then flattens; the flat part is the equilibrated portion.

    :param states: Output of :func:`state_series` (optionally merged).
    :param n_points: Number of start offsets to evaluate.
    :return: ``(start_fractions, p_folded)``, both shape ``(n_points,)``."""
    seq = np.asarray(states)
    n = len(seq)
    starts = np.linspace(0, int(0.9 * n), n_points).astype(int)
    out = []
    for s in starts:
        tail = seq[s:]
        valid = tail[tail >= 0]
        out.append(float(np.mean(valid == FOLDED)) if len(valid) else float("nan"))
    return starts / max(n, 1), np.array(out)


def cutoff_scan(
    cv_segments: list[np.ndarray],
    lo_grid: np.ndarray,
    hi_grid: np.ndarray,
    min_dwell_frames: int = 1,
) -> dict[str, np.ndarray]:
    """Transition count and folded population over a grid of core cutoffs.

    The point is to show the verdict does not hinge on one hand-picked pair: a robust
    result holds across a plateau of the grid. Entries where the cores overlap are NaN.

    :param cv_segments: One CV array per independent segment.
    :param lo_grid: Candidate ``q_unfolded`` values.
    :param hi_grid: Candidate ``q_folded`` values.
    :param min_dwell_frames: Passed to :func:`merge_events`.
    :return: ``{"lo", "hi", "transitions", "p_folded"}``; the 2-D arrays are
        ``(len(lo_grid), len(hi_grid))``."""
    transitions = np.full((len(lo_grid), len(hi_grid)), np.nan)
    populations = np.full((len(lo_grid), len(hi_grid)), np.nan)
    for i, lo in enumerate(lo_grid):
        for j, hi in enumerate(hi_grid):
            if lo >= hi:
                continue
            total, folded, assigned = 0, 0, 0
            for cv in cv_segments:
                st = merge_events(state_series(cv, lo, hi), min_dwell_frames)
                n_uf, n_fu = count_transitions(st)
                total += min(n_uf, n_fu)
                valid = st[st >= 0]
                folded += int(np.sum(valid == FOLDED))
                assigned += len(valid)
            transitions[i, j] = total
            populations[i, j] = folded / assigned if assigned else np.nan
    return {
        "lo": np.asarray(lo_grid),
        "hi": np.asarray(hi_grid),
        "transitions": transitions,
        "p_folded": populations,
    }


def msm_diagnostics(
    features: np.ndarray | list[np.ndarray],
    lags: list[int],
    n_clusters: int = 50,
    dt_ps: float = 5.0,
    seed: int = 0,
) -> dict[str, object]:
    """Implied timescales from a Markov state model, as a convergence *diagnostic*.

    Not used for any reported population or rate. Its job is to record whether a
    Markov model would have been trustworthy: timescales that keep growing with lag
    mean it would not, and that fact belongs in the output alongside the core-set
    numbers rather than being quietly omitted.

    :param features: Discretisation input, ``(n_frames, n_features)`` or a list of such.
    :param lags: Lag times in frames.
    :param n_clusters: k-means cluster count.
    :param dt_ps: Time between consecutive frames in ps.
    :param seed: RNG seed for k-means.
    :return: ``{"lags_ns", "timescales_ns", "converged", "note"}``; on failure
        ``converged`` is None and ``note`` carries the reason."""
    blocks = features if isinstance(features, list) else [features]
    try:
        from deeptime.clustering import KMeans
        from deeptime.markov.msm import MaximumLikelihoodMSM
    except ImportError as e:
        return {"lags_ns": [], "timescales_ns": [], "converged": None, "note": str(e)}

    try:
        clustering = KMeans(
            n_clusters=n_clusters, max_iter=50, fixed_seed=seed, progress=None
        ).fit_fetch(np.concatenate(blocks, axis=0))
        dtrajs = [clustering.transform(b) for b in blocks]
        lags_ns, timescales = [], []
        for lag in lags:
            if lag >= min(len(d) for d in dtrajs):
                continue
            msm = MaximumLikelihoodMSM(lagtime=lag).fit_fetch(dtrajs)
            ts = msm.timescales()
            lags_ns.append(lag * dt_ps / 1000.0)
            timescales.append(float(ts[0]) * dt_ps / 1000.0 if len(ts) else float("nan"))
    except Exception as e:  # deeptime raises a variety of estimation errors
        return {"lags_ns": [], "timescales_ns": [], "converged": None, "note": repr(e)}

    converged = None
    if len(timescales) >= 2 and np.isfinite(timescales[-2:]).all() and timescales[-2] > 0:
        # A plateau means the slowest timescale stops growing as the lag grows.
        converged = bool(abs(timescales[-1] - timescales[-2]) / timescales[-2] < 0.2)
    return {
        "lags_ns": lags_ns,
        "timescales_ns": timescales,
        "converged": converged,
        "note": "implied timescales; a plateau is required for MSM validity",
    }


def melting_curve(
    temperatures: np.ndarray, p_folded: np.ndarray, p_err: np.ndarray | None = None
) -> dict[str, object]:
    """Two-state van't Hoff fit of the folded fraction against temperature.

    Model ``dG(T) = dH (1 - T/T_m)`` with ``p_f = 1/(1 + exp(-dG/RT))``. Points at
    exactly 0 or 1 carry no information about the equilibrium constant (``ln K`` is
    infinite there) and are excluded from the fit and reported as censored bounds —
    without that, a fully folded run alone makes the fit diverge.

    :param temperatures: Temperatures in K.
    :param p_folded: Folded fraction at each temperature.
    :param p_err: Optional standard errors, used as fit weights.
    :return: ``{"t_m", "delta_h", "n_used", "censored", "indicative"}``; ``t_m`` and
        ``delta_h`` are None when fewer than three points are usable."""
    from scipy.optimize import curve_fit

    T = np.asarray(temperatures, dtype=float)
    p = np.asarray(p_folded, dtype=float)
    usable = np.isfinite(p) & (p > 0.0) & (p < 1.0)
    censored = [
        {"temperature": float(t), "p_folded": float(v)}
        for t, v in zip(T[~usable], p[~usable])
    ]
    if usable.sum() < 3:
        return {
            "t_m": None,
            "delta_h": None,
            "n_used": int(usable.sum()),
            "censored": censored,
            "indicative": True,
            "note": "fewer than three unsaturated points; no fit attempted",
        }

    def model(t, delta_h, t_m):
        return 1.0 / (1.0 + np.exp(-delta_h * (1.0 / t - 1.0 / t_m) / GAS_CONSTANT))

    sigma = None
    if p_err is not None:
        err = np.asarray(p_err, dtype=float)[usable]
        sigma = np.where(np.isfinite(err) & (err > 0), err, 1.0)
    popt, _ = curve_fit(
        model,
        T[usable],
        p[usable],
        p0=[50.0, float(np.mean(T))],
        sigma=sigma,
        maxfev=20000,
    )
    return {
        "t_m": float(popt[1]),
        "delta_h": float(popt[0]),
        "n_used": int(usable.sum()),
        "censored": censored,
        # Two parameters from a handful of points: a bracket, not a measurement.
        "indicative": bool(usable.sum() < 5),
        "note": "dG(T) = dH (1 - T/T_m); dH in kJ/mol, T_m in K",
    }


def analyse_run(
    cv_segments: list[np.ndarray],
    temperature: float,
    dt_ps: float,
    q_unfolded: float | None = None,
    q_folded: float | None = None,
    min_dwell_ps: float = 2000.0,
    n_blocks: int = 5,
    n_bootstrap: int = 2000,
    bins: int = 90,
    sigma: float = 2.0,
    x_range: tuple[float, float] | None = (0.0, 1.0),
    do_cutoff_scan: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    """Full basin analysis of one temperature, pooling independent segments.

    Populations pool across segments; transitions and dwell times never do — each
    segment is counted on its own so no seam becomes a transition.

    :param cv_segments: One CV array per independent segment.
    :param temperature: Temperature in K, for the record and the melting curve.
    :param dt_ps: Time between consecutive frames in ps.
    :param q_unfolded: Unfolded core edge; derived from the FES when None.
    :param q_folded: Folded core edge; derived from the FES when None.
    :param min_dwell_ps: Minimum dwell for a visit to count as an event.
    :param n_blocks: Blocks for the stationarity check.
    :param n_bootstrap: Moving-block bootstrap resamples.
    :param bins: Histogram bins for the FES.
    :param sigma: Gaussian smoothing width in bins.
    :param x_range: Histogram span; ``None`` uses the data extent.
    :param do_cutoff_scan: Also scan the cutoff grid (slower).
    :param seed: RNG seed.
    :return: A JSON-serialisable result dict."""
    pooled = np.concatenate(cv_segments)
    total_ns = len(pooled) * dt_ps / 1000.0
    result: dict[str, object] = {
        "temperature": float(temperature),
        "dt_ps": float(dt_ps),
        "n_segments": len(cv_segments),
        "n_frames": int(len(pooled)),
        "total_ns": float(total_ns),
        "cv_mean": float(pooled.mean()),
        "cv_std": float(pooled.std()),
        "cv_min": float(pooled.min()),
        "cv_max": float(pooled.max()),
    }

    # Always characterise the surface for the record. A surface that is not resolvably
    # two-state is a finding, not a failure: at low temperature the unfolded state shows
    # up as rare excursions rather than a separated well. Explicit cores let the kinetics
    # be measured anyway, which is the only way to quantify those excursions.
    try:
        cv_lo, cv_barrier, cv_hi = locate_basins(pooled, bins, sigma, x_range)
        derived_lo, derived_hi = core_cutoffs(pooled, bins, sigma, x_range)
        result |= {
            "cv_unfolded_min": cv_lo,
            "cv_barrier": cv_barrier,
            "cv_folded_min": cv_hi,
            "two_state": True,
        }
    except ValueError as e:
        derived_lo = derived_hi = None
        result |= {"two_state": False, "verdict": str(e)}

    q_unfolded = derived_lo if q_unfolded is None else q_unfolded
    q_folded = derived_hi if q_folded is None else q_folded
    if q_unfolded is None or q_folded is None:
        result["verdict"] = (
            f"{result.get('verdict', '')} No cores could be derived; pass "
            "--q-unfolded/--q-folded to measure the excursions anyway."
        ).strip()
        return result

    min_dwell_frames = max(1, int(round(min_dwell_ps / dt_ps)))
    result |= {
        "q_unfolded": float(q_unfolded),
        "q_folded": float(q_folded),
        "min_dwell_ps": float(min_dwell_ps),
        "min_dwell_frames": min_dwell_frames,
    }

    per_segment = [
        merge_events(state_series(cv, q_unfolded, q_folded), min_dwell_frames)
        for cv in cv_segments
    ]
    raw = [state_series(cv, q_unfolded, q_folded) for cv in cv_segments]

    n_uf = n_fu = raw_uf = raw_fu = 0
    folded_dwells, unfolded_dwells = [], []
    for merged, unmerged in zip(per_segment, raw):
        a, b = count_transitions(merged)
        n_uf += a
        n_fu += b
        a, b = count_transitions(unmerged)
        raw_uf += a
        raw_fu += b
        f, u = dwell_times(merged, dt_ps)
        folded_dwells.append(f)
        unfolded_dwells.append(u)
    folded_dwells = np.concatenate(folded_dwells) if folded_dwells else np.array([])
    unfolded_dwells = np.concatenate(unfolded_dwells) if unfolded_dwells else np.array([])

    pooled_states = np.concatenate(per_segment)
    p_folded, p_unfolded = basin_populations(pooled_states)
    tau_int, window = integrated_autocorr_time(pooled, dt_ps)
    n_eff = total_ns / (2.0 * tau_int) if tau_int > 0 else float("inf")
    sigma_p = (
        float(np.sqrt(max(p_folded * p_unfolded, 0.0) / n_eff))
        if np.isfinite(n_eff) and n_eff > 0
        else float("nan")
    )
    blocks = block_estimates(pooled_states, n_blocks)
    ci_low, ci_high = block_bootstrap(
        pooled_states, max(1, int(round(2 * tau_int * 1000.0 / dt_ps))), n_bootstrap, seed
    )
    rc_start, rc_pop = reverse_cumulative_population(pooled_states)

    result |= {
        "p_folded": p_folded,
        "p_unfolded": p_unfolded,
        "sigma_p": sigma_p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_events_unfolded_to_folded": n_uf,
        "n_events_folded_to_unfolded": n_fu,
        "n_round_trips": int(min(n_uf, n_fu)),
        "n_raw_crossings": int(raw_uf + raw_fu),
        "tau_int_ns": tau_int,
        "tau_window_frames": int(window),
        "n_eff": n_eff,
        "tau_fold_ns": float(unfolded_dwells.mean()) if len(unfolded_dwells) else None,
        "tau_unfold_ns": float(folded_dwells.mean()) if len(folded_dwells) else None,
        "n_folded_dwells": int(len(folded_dwells)),
        "n_unfolded_dwells": int(len(unfolded_dwells)),
        "block_p_folded": [None if np.isnan(v) else float(v) for v in blocks],
        "block_spread": float(np.nanmax(blocks) - np.nanmin(blocks)),
        "block_consistent_with_sigma": (
            bool(np.nanstd(blocks) <= 3.0 * sigma_p * np.sqrt(n_blocks))
            if np.isfinite(sigma_p)
            else None
        ),
        "reverse_cumulative_start": rc_start.tolist(),
        "reverse_cumulative_p_folded": [
            None if np.isnan(v) else float(v) for v in rc_pop
        ],
        "sampling_adequate": bool(
            min(n_uf, n_fu) >= 10 and min(p_folded, p_unfolded) >= 0.15
        ),
    }

    if do_cutoff_scan:
        grid = cutoff_scan(
            cv_segments,
            np.linspace(0.15, 0.45, 7),
            np.linspace(0.5, 0.8, 7),
            min_dwell_frames,
        )
        result["cutoff_scan"] = {k: np.asarray(v).tolist() for k, v in grid.items()}
    return result


def plot_basins(
    results: list[dict],
    cv_by_temperature: dict[float, list[np.ndarray]],
    out_file: Path,
    cv_label: str = r"Fraction of native contacts $Q$",
    bins: int = 90,
    sigma: float = 2.0,
    f_max: float = 7.0,
) -> None:
    """Three diagnostic panels per temperature into one PDF.

    Columns: the CV trace with the cores marked, the 1-D free energy, and the
    reverse-cumulative folded fraction. Read together they say whether both basins are
    visited, whether they are separated, and whether the populations have settled.

    :param results: Per-temperature dicts from :func:`analyse_run`.
    :param cv_by_temperature: The CV segments each result was computed from.
    :param out_file: Output ``.pdf`` path.
    :param cv_label: Axis label for the collective variable.
    :param bins: Histogram bins for the FES panel.
    :param sigma: Gaussian smoothing width in bins.
    :param f_max: Free-energy ceiling in kT for the FES panel."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature"])
    except (ImportError, OSError):
        pass

    n = len(results)
    fig, axes = plt.subplots(
        n, 3, figsize=(6.9, 1.8 * n), constrained_layout=True, squeeze=False
    )
    for row, res in enumerate(results):
        T = res["temperature"]
        segments = cv_by_temperature[T]
        pooled = np.concatenate(segments)
        dt_ns = res["dt_ps"] / 1000.0

        ax = axes[row][0]
        ax.plot(np.arange(len(pooled)) * dt_ns, pooled, lw=0.2, rasterized=True)
        for key, style in (("q_folded", "-"), ("q_unfolded", "--")):
            if res.get(key) is not None:
                ax.axhline(res[key], color="k", ls=style, lw=0.6)
        # Segment seams are where continuity in time breaks.
        edge = 0
        for seg in segments[:-1]:
            edge += len(seg)
            ax.axvline(edge * dt_ns, color="0.6", lw=0.4)
        ax.set_ylabel(f"{T:.0f} K\n{cv_label}")
        ax.set_xlabel("Time [ns]")

        ax = axes[row][1]
        centres, F = free_energy_1d(pooled, bins, sigma, f_max, (0.0, 1.0))
        ax.plot(centres, F)
        if res.get("cv_barrier") is not None:
            ax.axvline(res["cv_barrier"], color="k", ls=":", lw=0.6)
        ax.set_xlabel(cv_label)
        ax.set_ylabel(r"$F$ $[k_{\mathrm{B}}T]$")

        ax = axes[row][2]
        if res.get("reverse_cumulative_p_folded") is not None:
            start = np.asarray(res["reverse_cumulative_start"], dtype=float)
            pop = np.array(
                [np.nan if v is None else v for v in res["reverse_cumulative_p_folded"]]
            )
            ax.plot(start, pop)
            if res.get("p_folded") is not None and np.isfinite(res.get("sigma_p", np.nan)):
                ax.axhspan(
                    res["p_folded"] - res["sigma_p"],
                    res["p_folded"] + res["sigma_p"],
                    color="0.85",
                )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Discarded initial fraction")
        ax.set_ylabel(r"$p_{\mathrm{folded}}$")

    fig.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_melting_curve(results: list[dict], fit: dict, out_file: Path) -> None:
    """Folded fraction against temperature with the two-state fit overlaid.

    :param results: Per-temperature dicts from :func:`analyse_run`.
    :param fit: Output of :func:`melting_curve`.
    :param out_file: Output ``.pdf`` path."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature"])
    except (ImportError, OSError):
        pass

    T = np.array([r["temperature"] for r in results], dtype=float)
    p = np.array([r.get("p_folded", np.nan) for r in results], dtype=float)
    lo = np.array([r.get("ci_low", np.nan) for r in results], dtype=float)
    hi = np.array([r.get("ci_high", np.nan) for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(3.3, 2.5), constrained_layout=True)
    err = np.vstack([np.abs(p - lo), np.abs(hi - p)])
    ax.errorbar(T, p, yerr=np.where(np.isfinite(err), err, 0.0), fmt="o", capsize=2)
    if fit.get("t_m") is not None:
        grid = np.linspace(T.min() - 20, T.max() + 20, 200)
        dh, tm = fit["delta_h"], fit["t_m"]
        ax.plot(
            grid,
            1.0 / (1.0 + np.exp(-dh * (1.0 / grid - 1.0 / tm) / GAS_CONSTANT)),
            "-",
            label=rf"$T_m$ = {tm:.0f} K, $\Delta H$ = {dh:.0f} kJ/mol",
        )
        ax.axvline(tm, color="k", ls=":", lw=0.6)
        ax.legend(fontsize=6)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"$p_{\mathrm{folded}}$")
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close(fig)


def load_cv_segments(
    npz_path: Path, cv_name: str
) -> tuple[list[np.ndarray], float, dict[str, object]]:
    """Read one CV out of an ``md-sim-cvs`` npz, split into independent segments.

    :param npz_path: Path to a ``<prefix>_cvs.npz``.
    :param cv_name: Which CV array to read.
    :return: ``(segments, dt_ps, metadata)``.
    :raises KeyError: If the CV is absent from the file."""
    data = np.load(npz_path)
    if cv_name not in data:
        available = [k for k in data.files if not k.startswith(("stride", "dt_ps", "q_"))]
        raise KeyError(
            f"'{cv_name}' not in {npz_path}. Available CVs: {available}. "
            "Rerun md-sim-cvs with that CV requested."
        )
    cv = np.asarray(data[cv_name], dtype=float)
    dt_ps = float(data["dt_ps"]) if "dt_ps" in data else 1.0
    if "segment_lengths" in data:
        lengths = np.asarray(data["segment_lengths"], dtype=int)
    else:
        lengths = np.array([len(cv)])
    if lengths.sum() != len(cv):
        logger.warning(
            f"{npz_path}: segment_lengths sum to {lengths.sum()} but the CV has "
            f"{len(cv)} frames; treating it as one segment."
        )
        lengths = np.array([len(cv)])
    segments = np.split(cv, np.cumsum(lengths)[:-1])
    meta = {
        "npz": str(npz_path),
        "stride": int(data["stride"]) if "stride" in data else None,
        "q_min_seq_sep": int(data["q_min_seq_sep"]) if "q_min_seq_sep" in data else None,
        "segment_lengths": lengths.tolist(),
    }
    if "dt_ps" not in data:
        logger.warning(
            f"{npz_path} has no dt_ps; assuming 1 ps per frame. Rerun md-sim-cvs "
            "with --dt-ps so kinetics are in physical units."
        )
    return segments, dt_ps, meta


def main() -> None:
    """Entry point: basin populations, kinetics and convergence per temperature."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cvs-npz",
        required=True,
        nargs="+",
        type=Path,
        help="One md-sim-cvs npz per temperature.",
    )
    parser.add_argument(
        "--temperatures",
        required=True,
        nargs="+",
        type=float,
        help="Temperature in K for each --cvs-npz, in the same order.",
    )
    parser.add_argument(
        "--cv",
        default="native_contacts",
        help="Which CV in the npz to use as the folding coordinate.",
    )
    parser.add_argument(
        "--q-folded",
        type=float,
        default=None,
        help="Folded core edge; derived from the 1-D FES when omitted.",
    )
    parser.add_argument(
        "--q-unfolded",
        type=float,
        default=None,
        help="Unfolded core edge; derived from the 1-D FES when omitted.",
    )
    parser.add_argument(
        "--min-dwell-ps",
        type=float,
        default=2000.0,
        help="Minimum dwell for a basin visit to count as an event (default 2000).",
    )
    parser.add_argument("--n-blocks", type=int, default=5, help="Stationarity blocks.")
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000, help="Block-bootstrap resamples."
    )
    parser.add_argument("--bins", type=int, default=90, help="FES histogram bins.")
    parser.add_argument("--sigma", type=float, default=2.0, help="FES smoothing in bins.")
    parser.add_argument(
        "--f-max", type=float, default=7.0, help="FES plotting ceiling in kT."
    )
    parser.add_argument(
        "--cutoff-scan", action="store_true", help="Scan the core-cutoff grid."
    )
    parser.add_argument(
        "--msm-lags",
        type=int,
        nargs="*",
        default=None,
        help="Fit MSM implied timescales at these lags (frames) as a diagnostic.",
    )
    parser.add_argument(
        "--msm-clusters", type=int, default=50, help="k-means clusters for the MSM."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory.")
    parser.add_argument("--prefix", default="basins", help="Output filename prefix.")
    parser.add_argument("--plot", action="store_true", help="Also write the figures.")
    args = parser.parse_args()

    if len(args.cvs_npz) != len(args.temperatures):
        parser.error(
            f"Got {len(args.cvs_npz)} npz files but {len(args.temperatures)} "
            "temperatures; they must correspond one to one."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    cv_by_temperature: dict[float, list[np.ndarray]] = {}
    for npz_path, temperature in zip(args.cvs_npz, args.temperatures):
        segments, dt_ps, meta = load_cv_segments(npz_path, args.cv)
        res = analyse_run(
            segments,
            temperature,
            dt_ps,
            q_unfolded=args.q_unfolded,
            q_folded=args.q_folded,
            min_dwell_ps=args.min_dwell_ps,
            n_blocks=args.n_blocks,
            n_bootstrap=args.n_bootstrap,
            bins=args.bins,
            sigma=args.sigma,
            do_cutoff_scan=args.cutoff_scan,
            seed=args.seed,
        )
        res["source"] = meta
        res["cv"] = args.cv
        if args.msm_lags:
            res["msm"] = msm_diagnostics(
                [s.reshape(-1, 1) for s in segments],
                args.msm_lags,
                args.msm_clusters,
                dt_ps,
                args.seed,
            )
        results.append(res)
        cv_by_temperature[float(temperature)] = segments

        if "p_folded" in res:
            flag = "" if res.get("two_state") else "  [FES not resolvably two-state]"
            print(
                f"{temperature:6.1f} K  p_folded {res['p_folded']:.3f} "
                f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
                f"round trips {res['n_round_trips']:3d} "
                f"(raw crossings {res['n_raw_crossings']:4d})  "
                f"tau_int {res['tau_int_ns']:5.1f} ns  N_eff {res['n_eff']:6.1f}{flag}"
            )
        else:
            print(f"{temperature:6.1f} K  {res['verdict']}")

    fit = melting_curve(
        np.array([r["temperature"] for r in results]),
        np.array([r.get("p_folded", np.nan) for r in results]),
        np.array([r.get("sigma_p", np.nan) for r in results]),
    )
    if fit.get("t_m") is not None:
        print(
            f"\nvan't Hoff: T_m = {fit['t_m']:.1f} K, "
            f"dH = {fit['delta_h']:.1f} kJ/mol "
            f"({fit['n_used']} points"
            f"{', indicative' if fit['indicative'] else ''})"
        )
    else:
        print(f"\nvan't Hoff: {fit['note']}")

    payload = {"results": results, "melting_curve": fit, "cv": args.cv}
    json_path = out_dir / f"{args.prefix}_basins.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")

    if args.plot:
        label = (
            r"Fraction of native contacts $Q$"
            if args.cv == "native_contacts"
            else args.cv
        )
        fig_path = out_dir / f"{args.prefix}_basins.pdf"
        plot_basins(
            results,
            cv_by_temperature,
            fig_path,
            cv_label=label,
            bins=args.bins,
            sigma=args.sigma,
            f_max=args.f_max,
        )
        print(f"Wrote {fig_path}")
        melt_path = out_dir / f"{args.prefix}_melting.pdf"
        plot_melting_curve(results, fit, melt_path)
        print(f"Wrote {melt_path}")


if __name__ == "__main__":
    main()
