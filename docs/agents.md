# Agent contract

How an agent works with `md-sim` in a workspace (a repo with an `md-workspace.yaml`). The
surface is typed and discoverable, so nothing here has to be learned from source. An agent that
follows it authors and verifies; the user runs.

## 1. Enumerate

- `md-sim --help` lists the groups and commands: `run`, `check-job`, `mcp`, and the groups `config`,
  `prep`, `analysis`, `sampling`. `md-sim <group>` lists a group's verbs.
- `md-sim --json` prints the same surface as one JSON object. Every `md-sim` command takes `--json`
  and prints `{"ok": true, ...}` on success, or `{"ok": false, "error": {"type", "message"}}` with
  exit 1 (expected failure) or 2 (usage). Parse that, not the human text.
- With the `mcp` extra installed, `md-sim mcp` serves the same surface over stdio as six tools:
  `list_configs`, `describe_config`, `validate_config`, `config_schema`, `new_config`, `check_job`.
  A result is the `--json` envelope plus `"workspace"`.

## 2. Author against the schema

- `md-sim config schema [--engine E]` is the JSON Schema of the config; read it instead of the
  models. `md-sim config ls` lists the workspace's configs, `md-sim config describe <cfg>` shows
  what one resolves to (engine, system, `output_dir`, each input path with an `exists` flag,
  frames).
- Start from an existing config: `md-sim config new --from <cfg> --set k=v -o <out>` writes a
  validated copy (`k` may be dotted; an unknown field is an error). Write configs into the
  workspace's `config_dir` only.
- A config carries no queue facts: no partition, walltime or `slurm:` block. Those belong to the
  job script and the site profile.

## 3. Verify before handing over

1. `md-sim config validate <cfg>` exits 0, or 1 with every error as `{"loc", "msg", "type"}`.
2. Write the job script by hand, starting from a template example in the workspace's `slurm/`.
   Put the `#SBATCH` header before the first command.
3. `md-sim check-job <script.sh> [--require-gpu]` reports one line per check (`directives`,
   `config`, `inputs`, `partition`, `gres`, `output`, `venv`). Pass `--require-gpu` when the
   job needs a GPU: the config cannot say so, and without the flag a missing `--gres` is `skipped`,
   not a failure. A `skipped` check is a fact the script does not state statically, never a pass:
   state it in the script or tell the user.
4. Fix every `ok: false` and rerun until `check-job` exits 0.

## 4. Never

- **Never write into the data root.** It is the user's store (the lab repo's git-ignored `data/`);
  outputs, trajectories and checkpoints land there and nowhere else is touched. Read inputs from it, and point `output_subdir` at a fresh subdirectory rather
  than an existing run.
- **Never submit.** Do not call `sbatch`, and do not start `md-sim run` for anything longer than a
  smoke test. Hand the script to the user with its path and the `check-job` result; they submit it
  from a submit node. No MCP tool starts a simulation either.
- **Never commit** trajectories, checkpoints or the data root, and never edit `md-workspace.yaml`
  or `sites/` to make a check pass: a failing check names a real mismatch.
- Run state belongs to Beamtime (its CLI and MCP server); `md-sim` has no run listing.
