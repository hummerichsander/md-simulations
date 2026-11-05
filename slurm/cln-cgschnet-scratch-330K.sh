#!/bin/bash
#SBATCH --job-name=cln-cgschnet-scratch-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# CGSchNet force matching for chignolin at 330 K, from scratch, then the merge with the fitted
# priors into the model configs/cln-cgschnet-330K.yaml runs.
#
# Inputs, built once (see configs/training/cln-cgschnet-scratch-330K.yaml):
#   md-sim-cgschnet-dataset --run data/cln/AMBER14_implicit_330K_r0{0..7} \
#       --bead-pdb data/pdbs/1UAO_5B-CG.pdb --out data/cln/cln_fm_330K --name cln \
#       --start 2000 --fit-frames 2000
#   md-sim-cgschnet-priors --dataset data/cln/cln_fm_330K --scheme protein
#
# The priors are the transferable model's terms (`--scheme protein`): the generic bond-graph
# scheme gives the stiff omega / carbonyl / proline-ring dihedrals a 6-degree Fourier fit that
# blows up. They explain 85.9 % of the CG force variance. Batch 256 peaks at ~12 GB (3.1 GB at 64).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

# the training config, e.g. CONFIG=configs/training/cln-cgschnet-scratch-330K-long.yaml for the
# 150-epoch run; checkpoints and the merged model follow its default_root_dir.
CONFIG=${CONFIG:-configs/training/cln-cgschnet-scratch-330K.yaml}
ROOT=$(awk '/default_root_dir:/ {print $2; exit}' "${CONFIG}")
echo "config: ${CONFIG}, root: ${ROOT}"

mlcg-train_h5 fit --config "${CONFIG}"

# checkpoint names end in validation_loss=<value>.ckpt; keep the lowest.
BEST=$(ls "${ROOT}"/ckpt/cln-*.ckpt | sort -t= -k3 -g | head -n1)
echo "best checkpoint: ${BEST}"

mlcg-combine_model \
    --ckpt "${BEST}" \
    --prior data/cln/cln_fm_330K/priors.pt \
    --out "${ROOT}.pt"

echo "Finished; run CG MD with: sbatch slurm/cln-cgschnet-330K.sh"
