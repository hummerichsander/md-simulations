# Multiscale alanine dipeptide — a PyTorch starter

A self-contained introduction to the differentiable potentials in
`md_simulations.torch_potentials`, at two resolutions of the same molecule:

- **all-atom** — ACE-ALA-NME, 22 atoms, AMBER14 + GBn2 implicit solvent, evaluated through
  OpenMM;
- **coarse-grained** — 5 beads `C–N–CA–C–N`, a Boltzmann-inverted potential built from
  scratch in PyTorch.

Both obey the same contract: positions in **nm**, energy in **kJ/mol**, forces from
`backward()`.

## Install

Python 3.10–3.12, from a checkout of the `md-simulations` repository:

```bash
pip install -e "/path/to/md-simulations[torch-potentials]"
pip install -r requirements.txt
jupyter lab multiscale_ala2.ipynb
```

## What is in here

| file | what it is |
| --- | --- |
| `multiscale_ala2.ipynb` | the notebook |
| `ala2-amber14-implicit-330K.yaml` | the system the all-atom potential is built from |
| `data/pdbs/ala2.pdb` | the input structure the config points at |
| `data/ala2_topology.pdb` | the 22-atom topology, in the force field's atom order |
| `data/ala2_330K_xyz.npy` | 20 200 reference frames, `(20200, 22, 3)` in nm, same order |
| `prepare_data.py` | regenerates the two data files from the repository's trajectories |

## About the reference frames

Two choices behind them are worth knowing, because both are easy to get wrong and neither
announces itself:

**They come from replica exchange, not plain MD.** At 330 K a 10 ns plain trajectory spends
0.0004 % of its frames in the $\alpha_L$ basin — it is effectively missing. The frames here
are the 330 K rung of `configs/ala2-amber14-implicit-remd-330K-train.yaml` (101 ns of T-REMD,
same molecule and force field), which is a proper canonical ensemble at 330 K but crosses the
barrier often enough to populate it. The subsample keeps every 5th frame and is uniform in
time, so the Boltzmann weights the notebook inverts are unbiased:

| basin | population |
| --- | --- |
| C7eq / β | 63.7 % |
| α_R | 31.9 % |
| α_L | 1.7 % |
| C7ax | 0.3 % |

**They were relaxed into the unconstrained potential.** The production run was integrated with
`constraints: hbonds`, which holds all twelve X–H bonds at their ideal length *and removes
their energy term from the force field*. The notebook's config is `constraints: none`, so
those coordinates are back in the potential — and the constrained frames are not samples of
it. `prepare_data.py` therefore runs a short Langevin relaxation (250 fs at γ = 50 ps⁻¹,
0.5 fs steps) of every frame in the unconstrained potential: long enough for the X–H modes to
thermalise (their width goes from 0.0030 to 0.0043 nm, converged against a 60 ps free run),
short enough that the backbone torsions do not move — the basin populations above shift by
less than 0.03 %.

```bash
python prepare_data.py --repo /path/to/md-simulations     # ~2 min on 16 cores
```
