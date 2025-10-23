#!/bin/bash
#SBATCH -p milan7513_A100
#SBATCH --gres gpu:A100:2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
##SBATCH --begin=21:00:00
##SBATCH --time=120:00:00
#SBATCH --job-name=crps
#SBATCH --exclusive
#SBATCH -w pise35

# export WORKDIR="/home_nfs/audrena/py4cast"
export WORKDIR=$(pwd)
# export DATA="/tmp/TITAN"
# export DATA_S="/dataset/TITAN"
# export SINGULARITY_IMAGE="/scratch/audrena/py4cast2.sif"
export SINGULARITY_IMAGE="/scratch/audrena/editable_mfai.sif"
export PY4CAST_ROOTDIR="/scratch/audrena/py4cast_expe"
export PY4CAST_TITAN_PATH="/data/TITAN" #/fs1/arome/TITAN"
export http_proxy="http://10.197.209.6:3128"
export https_proxy="http://10.197.209.6:3128"
export CKPT_DIR="/scratch/audrena/py4cast_expe/titan_21var"
export PATH2CKPT="${CKPT_DIR}/afcrps_ARPdownScale_0order_1em4_CLN_4M_2BS/checkpoints/epoch=109-val_mean_loss=2.92.ckpt"
export NCCL_DEBUG=INFO



#--bind /scratch/audrena/py4cast_expe/titan_30var/Unetrpp_crps_4nm/checkpoints:/path2ckpt \

#--bind /home_nfs/audrena/py4cast/py4cast:/opt/conda/lib/python3.11/site-packages/py4cast \ once py4cast repo updated

time srun --mpi=pmix \
    singularity exec \
        --nv \
        --no-home \
        --writable-tmpfs \
        --bind ${PY4CAST_ROOTDIR} \
        --bind /scratch/audrena/py4cast_expe:/py4cast_root_dir \
        --bind /home_nfs/audrena/mfai/mfai:/opt/conda/lib/python3.11/site-packages/mfai \
        --bind ${PY4CAST_TITAN_PATH} \
        --bind ${WORKDIR}:/py4cast \
        --pwd /py4cast \
        ${SINGULARITY_IMAGE} \
        bash -c " \
            pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e . --no-cache-dir && \
            python bin/main.py test \
                -c config/CLI/trainer.yaml \
                -c config/CLI/dataset/titan.yaml \
                -c config/CLI/model/unetrpp.yaml \
        "