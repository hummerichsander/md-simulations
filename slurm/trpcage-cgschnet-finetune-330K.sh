#!/bin/bash
#SBATCH --job-name=trp-cage-cgschnet-finetune-330K
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# CGSchNet force matching on the 330 K trp-cage set (480,000 frames: 420,000 train over the
# seven seeded replicas, 12,000 validation from the held-out unseeded `equil` run, strided 5).
#
# Four arms, one per A30. The first run - batch 16 x 8, lr 1e-4, ReduceLROnPlateau,
# EarlyStopping patience 3 - took the held-out delta-force MSE from 46.68 (prior + pretrained
# SchNet) to 43.99, a 5.8% gain of which essentially all landed inside epoch 0. That single
# number is consistent with three quite different situations, and the sweep separates them:
#
#   0  ft-lr3e-4        pretrained init, 128 x 4  (324,480 par),  lr 3e-4
#   1  ft-lr1e-3        pretrained init, 128 x 4  (324,480 par),  lr 1e-3
#   2  scratch128       random init,     128 x 4  (324,480 par),  lr 1e-3
#   3  scratch256       random init,     256 x 6  (1,780,992 par), lr 1e-3
#
# Arms 0/1 test whether the first run merely stalled near its starting point: lr 1e-4 with an
# effective batch of 128 and a plateau reducer triggering on third-decimal noise. Arm 2 is the
# architectural control - same 324,480 parameters, fresh weights - so arm 2 against arms 0/1
# says whether the pretrained representation transfers at all on 420,000 frames of one
# protein. Arm 3 is the capacity probe.
#
# The outcome that matters for the downstream near-exact-PMF use is arm 3 against the rest. A
# direct estimate of the irreducible force-matching noise floor failed on this data - the
# closest frame pairs are 0.4-0.6 A RMSD apart, far too distant for a degenerate-pair
# estimator once delta forces are dominated by stiff local geometry - so four arms of
# different capacity and initialisation converging near 43.5-44 is how that floor gets bounded
# from above instead.
#
# Every arm runs 20 epochs of cosine-annealed Adam at an effective batch of 256 - 64 x 4 for
# the 128-wide arms, 16 x 16 for the wider one - so the four are compared at one gradient-noise
# level. Those native batches come from a measurement, not a guess: peak allocation is linear
# in the batch at 196 MB/sample for 128 x 4 and 553 MB/sample for 256 x 6, putting the arms at
# ~12.3 and ~8.7 GiB of the A30's 24. No EarlyStopping: annealing to eta_min is the point of
# the schedule, and stopping on a noisy validation metric defeats it. ModelCheckpoint keeps the
# best three epochs per arm.
#
# Runtime: the first run measured ~41 min/epoch at batch 16 on an 8 GB RTX 4060. The A30 has
# 3.4x the memory bandwidth, which is what a scatter-bound GNN is limited by, and batch 64
# improves utilisation, so ~20 min/epoch is the expectation - roughly 7 h for the 128-wide arms
# and longer for arm 3, which is both 5.5x the parameters and a quarter of the native batch.
# `max_time: 00:23:00:00` in each config is the backstop; if it trips, the arm stopped cleanly
# with `last.ckpt` written and the resume command is at the bottom of this file. Each task
# writes its own default_root_dir, so the four are independent and one failure costs only that
# arm.
#
# The result is the 330 K PMF and is valid only at 330 K: the CG PMF is a free energy,
# F(R;T) = U(R) - T S(R), whose mapping entropy is strongly R-dependent for a folding
# transition. Training set coverage is the 330 K Boltzmann ensemble, so ~90% folded - the fit
# is pointwise-unbiased where it has data and extrapolates in the expanded-Rg bands the
# transferable model's own ensemble visits.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"
# Activations for the 15 A radius graph are allocated and freed every step at a size that
# varies with the batch's edge count, which fragments the caching allocator over 20 epochs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=data/models/cgschnet/finetune_330K_a30
ARM_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "${ARM_ID}" in
  0)
    ARM=ft-lr3e-4
    CONFIG=configs/training/trpcage-cgschnet-finetune-330K-a30.yaml
    EXTRA=(--optimizer.init_args.lr 3.0e-4)
    ;;
  1)
    ARM=ft-lr1e-3
    CONFIG=configs/training/trpcage-cgschnet-finetune-330K-a30.yaml
    EXTRA=(--optimizer.init_args.lr 1.0e-3)
    ;;
  2)
    ARM=scratch128
    CONFIG=configs/training/trpcage-cgschnet-scratch128-330K-a30.yaml
    EXTRA=()
    ;;
  3)
    ARM=scratch256
    CONFIG=configs/training/trpcage-cgschnet-scratch256-330K-a30.yaml
    EXTRA=()
    ;;
  *)
    echo "unknown array task ${ARM_ID}; expected 0-3." >&2
    exit 1
    ;;
esac

echo "=== array task ${ARM_ID}: ${ARM} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --trainer.default_root_dir is overridden rather than baked into each config so all four arms
# land under one parent. mlcg's CLI substitutes it into the CSVLogger save_dir and the
# ModelCheckpoint dirpath after parsing, so the override propagates to both.
mlcg-train_h5 fit \
    --config "${CONFIG}" \
    --trainer.default_root_dir "${ROOT}/${ARM}" \
    "${EXTRA[@]}"

echo "Finished ${ARM}."

# Resume an arm that hit max_time or was preempted, e.g. for arm 0:
#
#   mlcg-train_h5 fit --config configs/training/trpcage-cgschnet-finetune-330K-a30.yaml \
#       --trainer.default_root_dir data/models/cgschnet/finetune_330K_a30/ft-lr3e-4 \
#       --optimizer.init_args.lr 3.0e-4 \
#       --ckpt_path data/models/cgschnet/finetune_330K_a30/ft-lr3e-4/ckpt/last.ckpt
#
# Then merge the winning arm's best checkpoint with the priors before simulating:
#
#   mlcg-combine_model --ckpt <root>/<arm>/ckpt/<best>.ckpt \
#       --prior data/trp-cage/fm_330K/prior_only.pt \
#       --out data/models/cgschnet/trpcage_finetuned_330K.pt
