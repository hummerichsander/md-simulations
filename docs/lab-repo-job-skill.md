Draft skill for the lab repo `/remote/gpu04/hummerich/projects/simulations`: copy everything from the
`---` line below to the end into `.claude/skills/write-job/SKILL.md` there (D2 of
`docs/plans/split-package-and-lab.md`). Delete this paragraph when pasting.

**Committing the skill.** The lab repo inherits the package's `.gitignore`, which ignores `.claude`
(and `.mcp.json`, deliberately: it is local to each checkout). With `.claude` ignored the skill cannot
be committed. Narrow the rule to what is really local, in the lab repo:

```bash
sed -i 's|^\.claude$|.claude/settings.local.json|' .gitignore
git add .gitignore .claude/skills/write-job/SKILL.md CLAUDE.md
```

`scripts/split-lab-repo.sh` does not make this edit; do it by hand after the split.

---
name: write-job
description: Write an sbatch script for a config in this lab repo and verify it with md-sim check-job. Use when asked to write, adapt or fix a Slurm job script, or to prepare a simulation for submission.
---

# Write a job script

Prepare a script the user can submit. The loop ends at a script that passes `md-sim check-job` and a
handover; it never ends in a submission. `sbatch` does not exist on gpu04 and is the user's to run.

1. **Have a valid config.** The script runs `md-sim run configs/<name>.yaml`, so the config comes
   first: `md-sim config validate configs/<name>.yaml` must exit 0. Author or copy one as `CLAUDE.md`
   describes (`md-sim config schema`, `md-sim config new --from`). `md-sim config describe` shows
   the resolved `output_dir`, each input path with an `exists` flag and the frame count.
2. **Read the site profile.** Open `sites/<site>.yaml` (the `site:` named in `md-workspace.yaml`).
   It lists the valid `partitions`, the `venv_activate` line the script must source, the `slurm_output`
   directory for `--output`/`--error`, and any `modules` to load. Take these values from the file;
   never from memory or from another script that may predate it.
3. **Copy the nearest template.** Pick from `slurm/*.sh.example` by engine family; if none fits, take the
   closest real script in `slurm/` (a replica array, a sharded array job, a chained pair). Copy it to
   `slurm/<name>.sh`, without the `.example` suffix. Do not edit the `.example` file.
4. **Write the script.** Change only what the job changes:
   - the `#SBATCH` header, before the first command: `--job-name`, `--time`, `--mem`, `--partition`
     from the site profile, `--output` and `--error` under `slurm_output`, and `--gres=gpu:1` when the
     job needs a GPU (any OpenMM job on the CUDA platform, any `cgschnet` job). The walltime is yours to
     estimate from the config's `describe` output and a comparable script's `--time`; say so in the
     handover when it is a guess.
   - the venv activation and `mkdir -p` of the log directory, as in the template;
   - the `md-sim run` line(s), pointing at the config. A script that runs several configs lists every
     one on a plain `md-sim run configs/...` line `check-job` can read; if a config path is built at
     run time (an array index, a variable), `check-job` reports it `skipped`, so say so.
   - `export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"`, or whatever the template does. Never point
     the data root, or a config's `output_subdir`, at an existing run.
5. **Check.** `md-sim check-job slurm/<name>.sh`, adding `--require-gpu` when the job needs a GPU
   (the config cannot say so; without the flag a missing `--gres` is only `skipped`). `md-sim check-job
   --help` has the flags and the checks. Fix every failing check in the script and rerun until it exits 0.
   Never edit `md-workspace.yaml` or `sites/` to make a check pass. A `skipped` check is a fact the
   script does not state statically, not a pass: state it in the script, or name it in the handover.
6. **Hand over; do not submit.** Reply with: the script path, the config path, the `check-job` result
   (including every `skipped`), what you guessed (walltime, memory), whether the run produces a
   dataset meant for the shared root (that needs a `datacard.yaml`), and the submit command as a standalone line
   for the user to run from a submit node:

   ```
   sbatch --chdir=/remote/gpu04/hummerich/projects/simulations slurm/<name>.sh
   ```

   Variables go in `sbatch --export=ALL,VAR=value ...`, never in front of the command. Do not run it,
   and do not try `squeue` or `scancel` either: none of them exist on gpu04.

## Never

- Call `sbatch`, or write a wrapper that does.
- Write into the data root (`data/`), or start `md-sim run` for more than a few smoke-test steps.
- Put a partition, walltime or `slurm:` block in a config: queue facts belong to the script and the
  site profile.
