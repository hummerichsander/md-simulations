Draft text for `/remote/gpu04/hummerich/projects/claude-instructions/workspace.md`. Paste it into the
bullet list of the "Workspace" section (replacing the existing `md-simulations` bullet), then delete
this paragraph. The user edits that file; a ticket session does not.

- `md-simulations` is the package other projects import: the MD engines, the typed config models,
  the `md-sim` command and the torch potentials other projects reweight with. It holds no configs,
  job scripts or data. `simulations` holds those: the configs, the hand-written job scripts and the
  Beamtime store, split out of `md-simulations` on 2026-11-26 (a full clone with its history, then
  each half deleted its other side). It reaches the shared data root through a `data/` symlink and
  finds its layout through `md-workspace.yaml`.
