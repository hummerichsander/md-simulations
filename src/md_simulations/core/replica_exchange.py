"""Pure bookkeeping of temperature replica exchange: ladder, swap criterion, statistics.

Nothing here touches OpenMM; :mod:`md_simulations.core.remd` drives the simulations with it.

Swap criterion
--------------
Two neighbouring temperatures ``i`` and ``j`` (inverse temperatures ``beta_i``, ``beta_j``) hold
configurations with potential energies ``u_i`` and ``u_j``. Exchanging them changes the joint
Boltzmann weight by ``exp((beta_i - beta_j) * (u_i - u_j))``, so the swap is accepted with
probability ``min(1, exp((beta_i - beta_j) * (u_i - u_j)))``. This satisfies detailed balance for
the product ensemble, provided the velocities are rescaled by ``sqrt(T_new / T_old)`` with the
move.

Exchanges alternate between the even pairs ``(0, 1), (2, 3), ...`` and the odd pairs
``(1, 2), (3, 4), ...``; the pairs of one parity are disjoint, so they can all be attempted at
once.
"""

from dataclasses import dataclass, field
from typing import Literal
import math

import numpy as np

Label = Literal["up", "down"] | None


def geometric_ladder(t_min: float, t_max: float, n: int) -> list[float]:
    """Build a geometric temperature ladder, which gives a roughly constant swap rate.

    :param t_min: Lowest temperature in Kelvin.
    :param t_max: Highest temperature in Kelvin.
    :param n: Number of temperatures, at least 2.
    :return: The ladder, with exactly ``t_min`` and ``t_max`` as its ends."""
    if n < 2:
        raise ValueError("a ladder needs at least 2 temperatures")
    ladder = [t_min * (t_max / t_min) ** (k / (n - 1)) for k in range(n)]
    ladder[-1] = t_max
    return ladder


def swap_probability(beta_i: float, beta_j: float, u_i: float, u_j: float) -> float:
    """Metropolis probability of exchanging the configurations at two temperatures.

    :param beta_i: Inverse temperature of the first slot.
    :param beta_j: Inverse temperature of the second slot.
    :param u_i: Potential energy of the configuration currently at ``beta_i``.
    :param u_j: Potential energy of the configuration currently at ``beta_j``.
    :return: ``min(1, exp((beta_i - beta_j) * (u_i - u_j)))``."""
    log_p = (beta_i - beta_j) * (u_i - u_j)
    # exp of a non-positive number cannot overflow, and underflows to 0.
    return 1.0 if log_p >= 0.0 else math.exp(log_p)


def attempted_pairs(parity: int, n: int) -> list[int]:
    """List the lower index ``k`` of every pair ``(k, k + 1)`` attempted at a parity.

    :param parity: 0 for the even pairs, 1 for the odd ones.
    :param n: Number of temperatures.
    :return: The lower indices."""
    return list(range(parity, n - 1, 2))


def attempt_swaps(
    energies: list[float],
    betas: list[float],
    replica_at: list[int],
    parity: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Attempt the neighbour swaps of one parity.

    One uniform number is drawn per attempted pair, whether or not the swap is certain, so a seeded
    run consumes the same random stream whatever the energies are.

    :param energies: ``energies[k]`` is the potential energy of the configuration at temperature
        ``k``.
    :param betas: Inverse temperature of every slot, ascending in temperature.
    :param replica_at: ``replica_at[k]`` is the replica currently at temperature ``k``.
    :param parity: 0 to attempt the even pairs, 1 for the odd ones.
    :param rng: The exchange random number generator.
    :return: The new ``replica_at`` (a permutation of the old one) and the accepted pairs, each
        given by its lower index ``k``."""
    new_replica_at = list(replica_at)
    accepted = []
    for k in attempted_pairs(parity, len(betas)):
        p = swap_probability(betas[k], betas[k + 1], energies[k], energies[k + 1])
        if rng.random() < p:
            new_replica_at[k], new_replica_at[k + 1] = new_replica_at[k + 1], new_replica_at[k]
            accepted.append(k)
    return new_replica_at, accepted


@dataclass
class ExchangeStats:
    """Swap acceptance per neighbour pair and replica round trips.

    A replica is labelled ``"up"`` when it last touched the lowest temperature and ``"down"`` when
    it last touched the highest after that. A round trip is a replica labelled ``"down"`` arriving
    at the lowest temperature again: lowest → highest → lowest.

    :param n: Number of temperatures."""

    n: int
    attempts: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    labels: dict[int, Label] = field(default_factory=dict)
    round_trips: int = 0

    def __post_init__(self) -> None:
        if not self.attempts:
            self.attempts = [0] * (self.n - 1)
            self.accepted = [0] * (self.n - 1)

    def touch_ends(self, replica_at: list[int]) -> None:
        """Update labels and the round-trip count from the replicas at the two ends.

        A replica only turns ``"down"`` if it came from the lowest temperature, so one that starts
        in the middle or on top does not count a round trip it never made.

        :param replica_at: ``replica_at[k]`` is the replica currently at temperature ``k``."""
        if self.labels.get(replica_at[-1]) == "up":
            self.labels[replica_at[-1]] = "down"
        lowest = replica_at[0]
        if self.labels.get(lowest) == "down":
            self.round_trips += 1
        self.labels[lowest] = "up"

    def record(self, parity: int, accepted: list[int], replica_at: list[int]) -> None:
        """Record one exchange attempt.

        :param parity: The parity attempted.
        :param accepted: Lower indices of the accepted pairs.
        :param replica_at: The arrangement after the exchange."""
        for k in attempted_pairs(parity, self.n):
            self.attempts[k] += 1
        for k in accepted:
            self.accepted[k] += 1
        self.touch_ends(replica_at)

    def acceptance(self, k: int) -> float:
        """Running acceptance of the pair ``(k, k + 1)``.

        :param k: Lower index of the pair.
        :return: Accepted over attempted, 0 before any attempt."""
        return self.accepted[k] / self.attempts[k] if self.attempts[k] else 0.0

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible types.

        :return: The state."""
        return {
            "n": self.n,
            "attempts": self.attempts,
            "accepted": self.accepted,
            "labels": {str(r): label for r, label in self.labels.items()},
            "round_trips": self.round_trips,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "ExchangeStats":
        """Restore from :meth:`to_dict`.

        :param state: The serialised state.
        :return: The statistics."""
        return cls(
            n=state["n"],
            attempts=list(state["attempts"]),
            accepted=list(state["accepted"]),
            labels={int(r): label for r, label in state["labels"].items()},
            round_trips=state["round_trips"],
        )
