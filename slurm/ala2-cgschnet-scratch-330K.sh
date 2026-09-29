#!/bin/bash
#SBATCH --job-name=ala2-cgschnet-scratch-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=gshort
#SBATCH --exclude=gpu05
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# CGSchNet force matching for alanine dipeptide at 330 K, from scratch, then the merge with
# the fitted priors into the model configs/ala2-cgschnet-330K.yaml runs.
#
# Inputs, built once (see configs/training/ala2-cgschnet-scratch-330K.yaml):
#   md-sim-cgschnet-dataset --run data/alaX/ala2_AMBER14_implicit_330K \
#       --bead-pdb data/pdbs/ala2_6B-CG.pdb --out data/alaX/ala2_fm_330K --name ala2 --stride 10
#   md-sim-cgschnet-priors --dataset data/alaX/ala2_fm_330K
#
# 450,000 training frames in batches of 512; ~0.15 s/step on an RTX 4060, so ~2 h for the 50
# epochs. gpu05's GTX 1080 Ti is sm_61, which the torch build here does not compile for.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

ROOT=data/models/cgschnet/ala2_scratch_330K

mlcg-train_h5 fit --config configs/training/ala2-cgschnet-scratch-330K.yaml

# checkpoint names end in validation_loss=<value>.ckpt; keep the lowest.
BEST=$(ls "${ROOT}"/ckpt/ala2-*.ckpt | sort -t= -k3 -g | head -n1)
echo "best checkpoint: ${BEST}"

mlcg-combine_model \
    --ckpt "${BEST}" \
    --prior data/alaX/ala2_fm_330K/priors.pt \
    --out data/models/cgschnet/ala2_scratch_330K.pt

echo "Finished; run CG MD with: sbatch slurm/ala2-cgschnet-330K.sh"
