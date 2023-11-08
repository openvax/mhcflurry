#!/bin/bash
#BSUB -J MHCf-{work_item_num} # Job name
#BSUB -P acc_nkcancer # allocation account or Unix group
#BSUB -q gpu # queue
#BSUB -gpu "num=1:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -R span[hosts=1] # one node
#BSUB -n 1 # number of compute cores
#BSUB -W 46:00 # walltime in HH:MM
#BSUB -R rusage[mem=30000] # mb memory requested
#BSUB -o {work_dir}/%J.stdout # output log (%J : JobID)
#BSUB -eo {work_dir}/STDERR # error log
#BSUB -L /bin/bash # Initialize the execution environment
#

set -e
set -x

echo "Subsequent stderr output redirected to stdout" >&2
exec 2>&1

export TMPDIR=/local/JOBS/mhcflurry-{work_item_num}
export PATH=$HOME/mhcflurry-conda-environment/bin/:$PATH
export PYTHONUNBUFFERED=1
export KMP_SETTINGS=1
#export TF_GPU_ALLOCATOR=cuda_malloc_async

free -m

module add cuda/11.8.0 cudnn/8.9.5-11
module list

nvidia-smi

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
#export XLA_FLAGS='--xla_compile=False'

python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'

env

cd {work_dir}

