#!/bin/bash
#SBATCH --job-name=alaX-cgschnet-scratch-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# CGSchNet force matching for one polyalanine at 330 K, from scratch, then the merge with the
# fitted priors into the model configs/${SYSTEM}-cgschnet-330K.yaml runs.
#
#   SYSTEM=ala8 sbatch --job-name=ala8-cgschnet-scratch-330K slurm/alaX-cgschnet-scratch-330K.sh
#
# Inputs, built once per system (n = 3 here; the CG pdb is split-flows' data/ala3-ms/top_cg.pdb):
#   md-sim-cgschnet-dataset --run data/alaX/ala3_AMBER14_implicit_330K \
#       --bead-pdb data/pdbs/ala3_14B-CG.pdb --out data/alaX/ala3_fm_330K --name ala3 \
#       --start 1000000 --stride 10 --fit-frames 2000 --write-cg-dcd
#   md-sim-cgschnet-priors --dataset data/alaX/ala3_fm_330K --n-blocks 8

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

SYSTEM=${SYSTEM:?set SYSTEM, e.g. SYSTEM=ala8}
CONFIG=${CONFIG:-configs/training/${SYSTEM}-cgschnet-scratch-330K.yaml}
ROOT=$(awk '/default_root_dir:/ {print $2; exit}' "${CONFIG}")
echo "system: ${SYSTEM}, config: ${CONFIG}, root: ${ROOT}"

mlcg-train_h5 fit --config "${CONFIG}"

# checkpoint names end in validation_loss=<value>.ckpt; keep the lowest.
BEST=$(ls "${ROOT}"/ckpt/"${SYSTEM}"-*.ckpt | sort -t= -k3 -g | head -n1)
echo "best checkpoint: ${BEST}"

mlcg-combine_model \
    --ckpt "${BEST}" \
    --prior "data/alaX/${SYSTEM}_fm_330K/priors.pt" \
    --out "${ROOT}.pt"

echo "Finished; run CG MD with: SYSTEM=${SYSTEM} sbatch slurm/alaX-cgschnet-330K.sh"
