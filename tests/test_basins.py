import numpy as np
import pytest

pytest.importorskip("scipy")

from md_simulations.analysis.basins import (
    basin_populations,
    block_bootstrap,
    block_estimates,
    core_cutoffs,
    count_transitions,
    cutoff_scan,
    dwell_times,
    integrated_autocorr_time,
    load_cv_segments,
    locate_basins,
    melting_curve,
    merge_events,
    reverse_cumulative_population,
    state_series,
)
from md_simulations.analysis.fes import free_energy_1d


def _bimodal(n_unfolded: int = 3000, n_folded: int = 7000, seed: int = 0) -> np.ndarray:
    """A two-state CV with minima planted at 0.20 and 0.85."""
    rng = np.random.default_rng(seed)
    return np.r_[
        rng.normal(0.20, 0.04, n_unfolded), rng.normal(0.85, 0.04, n_folded)
    ].clip(0.0, 1.0)


def _two_state_chain(n: int, k_uf: float, k_fu: float, seed: int = 0) -> np.ndarray:
    """A binary Markov chain with per-frame switch probabilities k_uf / k_fu."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int8)
    state = 1
    for i in range(n):
        p = k_fu if state == 1 else k_uf
        if rng.random() < p:
            state = 1 - state
        out[i] = state
    return out


def _ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    """An AR(1) series, whose integrated autocorrelation time is (1+phi)/(1-phi)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_free_energy_1d_recovers_planted_minima():
    centres, F = free_energy_1d(_bimodal(), bins=90, sigma=2.0, f_max=np.inf)
    minima = centres[
        np.r_[False, (F[1:-1] < F[:-2]) & (F[1:-1] < F[2:]), False]
    ]
    assert any(abs(m - 0.20) < 0.05 for m in minima)
    assert any(abs(m - 0.85) < 0.05 for m in minima)


def test_free_energy_1d_f_max_inf_keeps_surface():
    _, ceilinged = free_energy_1d(_bimodal(), 90, 2.0, 3.0, (0.0, 1.0))
    _, full = free_energy_1d(_bimodal(), 90, 2.0, np.inf, (0.0, 1.0))
    assert np.isnan(ceilinged).sum() > 0
    assert np.isnan(full).sum() == 0


def test_free_energy_1d_is_warning_free_on_empty_bins():
    # Empty bins would make a naive -log(p) emit a divide-by-zero warning.
    x = np.r_[np.zeros(500), np.ones(500)]
    with np.errstate(all="raise"):
        free_energy_1d(x, bins=90, sigma=0.0, f_max=np.inf, x_range=(0.0, 1.0))


def test_locate_basins_finds_minima_and_barrier():
    lo, barrier, hi = locate_basins(_bimodal())
    assert abs(lo - 0.20) < 0.05
    assert abs(hi - 0.85) < 0.05
    assert lo < barrier < hi


def test_locate_basins_raises_on_unimodal():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="not|two-state|separated"):
        locate_basins(rng.normal(0.9, 0.02, 5000).clip(0, 1))


def test_core_cutoffs_bracket_the_barrier():
    q_unfolded, q_folded = core_cutoffs(_bimodal())
    lo, barrier, hi = locate_basins(_bimodal())
    assert lo <= q_unfolded < q_folded <= hi
    assert q_unfolded < barrier < q_folded


def test_state_series_hysteresis_suppresses_intermediate_dithering():
    # Dithering strictly between the cores must never produce a transition.
    cv = np.array([0.9, 0.5, 0.55, 0.45, 0.5, 0.58, 0.42, 0.9])
    assert count_transitions(state_series(cv, 0.25, 0.6)) == (0, 0)


def test_state_series_marks_prefix_unassigned():
    states = state_series(np.r_[np.full(5, 0.4), np.full(5, 0.9)], 0.25, 0.6)
    assert (states[:5] == -1).all()
    assert (states[5:] == 1).all()


def test_state_series_rejects_overlapping_cores():
    with pytest.raises(ValueError, match="overlap"):
        state_series(np.linspace(0, 1, 10), 0.7, 0.3)


def test_real_excursion_counts_once():
    cv = np.r_[np.full(10, 0.9), np.full(10, 0.1), np.full(10, 0.9)]
    assert count_transitions(state_series(cv, 0.25, 0.6)) == (1, 1)


def test_merge_events_collapses_recrossings_of_one_excursion():
    # One 60-frame excursion, finely fragmented by jitter back across the folded core.
    cv = np.full(600, 0.9)
    cv[200:260] = 0.1
    cv[np.arange(205, 255, 4)] = 0.7
    raw = state_series(cv, 0.25, 0.6)
    assert sum(count_transitions(raw)) > 20  # the overcount being corrected

    merged = merge_events(raw, min_dwell_frames=10)
    # The excursion survives as exactly one event rather than being erased.
    assert count_transitions(merged) == (1, 1)


def test_merge_events_is_identity_below_two_frames():
    states = _two_state_chain(200, 0.05, 0.05)
    assert np.array_equal(merge_events(states, 1), states)


def test_dwell_times_drop_censored_first_and_last_visits():
    states = np.r_[
        np.ones(7), np.zeros(20), np.ones(30), np.zeros(20), np.ones(5)
    ].astype(np.int8)
    folded, unfolded = dwell_times(states, dt_ps=1000.0)
    assert np.allclose(folded, [30.0])
    assert np.allclose(unfolded, [20.0, 20.0])


def test_populations_and_mfpt_match_the_planted_chain():
    k_uf, k_fu = 0.02, 0.01
    states = _two_state_chain(400_000, k_uf, k_fu, seed=3)
    p_folded, p_unfolded = basin_populations(states)
    # Stationary distribution of the two-state chain.
    assert abs(p_folded - k_uf / (k_uf + k_fu)) < 0.02
    assert abs(p_folded + p_unfolded - 1.0) < 1e-12

    folded, unfolded = dwell_times(states, dt_ps=1000.0)
    # Mean dwell is the inverse switch probability, in frames -> ns at 1 ns/frame.
    assert abs(folded.mean() - 1.0 / k_fu) < 0.1 * (1.0 / k_fu)
    assert abs(unfolded.mean() - 1.0 / k_uf) < 0.1 * (1.0 / k_uf)


def test_integrated_autocorr_time_recovers_ar1_timescale():
    phi = 0.9
    tau, window = integrated_autocorr_time(_ar1(200_000, phi), dt_ps=1000.0)
    assert abs(tau - (1 + phi) / (1 - phi)) < 2.0
    assert window > 0


def test_integrated_autocorr_time_of_constant_series_is_zero():
    tau, _ = integrated_autocorr_time(np.full(1000, 0.5), dt_ps=5.0)
    assert tau == 0.0


def test_block_estimates_flag_a_non_stationary_series():
    drifting = np.r_[np.ones(500), np.zeros(500)].astype(np.int8)
    blocks = block_estimates(drifting, n_blocks=5)
    assert np.nanmax(blocks) - np.nanmin(blocks) > 0.9

    stationary = _two_state_chain(20_000, 0.05, 0.05, seed=1)
    blocks = block_estimates(stationary, n_blocks=5)
    assert np.nanmax(blocks) - np.nanmin(blocks) < 0.2


def test_block_bootstrap_brackets_the_point_estimate():
    states = _two_state_chain(20_000, 0.02, 0.01, seed=2)
    p_folded, _ = basin_populations(states)
    lo, hi = block_bootstrap(states, block_frames=200, n_resamples=400, seed=0)
    assert lo <= p_folded <= hi
    assert hi - lo > 0.0


def test_reverse_cumulative_population_flattens_after_a_burn_in():
    # 200 frames of pure folded burn-in, then a stationary 50/50 chain.
    tail = _two_state_chain(20_000, 0.1, 0.1, seed=4)
    states = np.r_[np.ones(2000, dtype=np.int8), tail]
    start, pop = reverse_cumulative_population(states, n_points=40)
    assert pop[0] > pop[-1]  # discarding the burn-in lowers p_folded
    assert abs(pop[-1] - 0.5) < 0.1


def test_segments_are_independent_no_seam_transition():
    ends_unfolded = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    starts_folded = np.full(100, 0.9)
    per_segment = sum(
        sum(count_transitions(merge_events(state_series(s, 0.25, 0.6), 5)))
        for s in (ends_unfolded, starts_folded)
    )
    spliced = sum(
        count_transitions(
            merge_events(
                state_series(np.r_[ends_unfolded, starts_folded], 0.25, 0.6), 5
            )
        )
    )
    assert per_segment == 1  # only the genuine unfolding in the first segment
    assert spliced == 2  # splicing invents a refolding at the seam


def test_cutoff_scan_reports_nan_where_cores_overlap():
    grid = cutoff_scan([_bimodal()], np.array([0.2, 0.7]), np.array([0.6, 0.8]))
    assert np.isnan(grid["transitions"][1, 0])  # lo=0.7 above hi=0.6
    assert np.isfinite(grid["transitions"][0, 0])


def test_cutoff_scan_transitions_grow_as_the_unfolded_core_widens():
    # Raising q_unfolded widens the unfolded core, so more excursions reach it and the
    # transition count can only go up. A verdict that flips across the grid is not robust.
    rng = np.random.default_rng(6)
    chain = _two_state_chain(20_000, 0.02, 0.02, seed=5)
    cv = np.where(chain == 1, 0.85, 0.20) + rng.normal(0.0, 0.04, len(chain))
    grid = cutoff_scan([cv], np.array([0.10, 0.18, 0.26]), np.array([0.80]))
    counts = grid["transitions"][:, 0]
    assert np.all(np.diff(counts) >= 0), counts
    assert counts[-1] > counts[0]


def test_melting_curve_recovers_planted_parameters():
    from md_simulations.analysis.basins import GAS_CONSTANT

    T = np.array([300.0, 320.0, 340.0, 360.0, 380.0])
    t_m, delta_h = 350.0, 60.0
    p = 1.0 / (1.0 + np.exp(-delta_h * (1.0 / T - 1.0 / t_m) / GAS_CONSTANT))
    fit = melting_curve(T, p)
    assert abs(fit["t_m"] - t_m) < 2.0
    assert abs(fit["delta_h"] - delta_h) < 2.0
    assert fit["censored"] == []


def test_melting_curve_excludes_saturated_points():
    from md_simulations.analysis.basins import GAS_CONSTANT

    T = np.array([300.0, 330.0, 360.0, 390.0])
    t_m, delta_h = 370.0, 60.0
    p = 1.0 / (1.0 + np.exp(-delta_h * (1.0 / T - 1.0 / t_m) / GAS_CONSTANT))
    p[0] = 1.0  # a fully folded run carries no information about ln K
    fit = melting_curve(T, p)
    assert fit["n_used"] == 3
    assert len(fit["censored"]) == 1
    assert abs(fit["t_m"] - t_m) < 5.0


def test_melting_curve_declines_to_fit_too_few_points():
    fit = melting_curve(np.array([300.0, 330.0]), np.array([1.0, 0.5]))
    assert fit["t_m"] is None
    assert fit["indicative"] is True


def test_load_cv_segments_splits_on_segment_lengths(tmp_path):
    path = tmp_path / "cv_cvs.npz"
    np.savez(
        path,
        native_contacts=np.arange(30, dtype=float),
        dt_ps=np.float64(5.0),
        stride=np.int64(1),
        segment_lengths=np.array([10, 20], dtype=np.int64),
    )
    segments, dt_ps, meta = load_cv_segments(path, "native_contacts")
    assert [len(s) for s in segments] == [10, 20]
    assert dt_ps == 5.0
    assert meta["segment_lengths"] == [10, 20]


def test_load_cv_segments_raises_for_missing_cv(tmp_path):
    path = tmp_path / "cv_cvs.npz"
    np.savez(path, rg=np.zeros(5), dt_ps=np.float64(5.0))
    with pytest.raises(KeyError, match="native_contacts"):
        load_cv_segments(path, "native_contacts")
