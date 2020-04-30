#!/bin/bash
#BSUB -J MHCf-{work_item_num} # Job name
#BSUB -P acc_nkcancer # allocation account or Unix group
#BSUB -q gpu # queue
#BSUB -R rusage[ngpus_excl_p=1]  # 1 exclusive GPU
#BSUB -R span[hosts=1] # one node
#BSUB -n 1 # number of compute cores
#BSUB -W 10:00 # walltime in HH:MM
#BSUB -R rusage[mem=20000] # mb memory requested
#BSUB -o {work_dir}/%J.stdout # output log (%J : JobID)
#BSUB -eo {work_dir}/STDERR # error log
#BSUB -L /bin/bash # Initialize the execution environment
#

set -e
set -x

echo "Subsequent stderr output redirected to stdout" >&2
exec 2>&1

export TMPDIR=/local/JOBS/mhcflurry-{work_item_num}
export PATH=$HOME/.conda/envs/py36b/bin/:$PATH
export PYTHONUNBUFFERED=1
export KMP_SETTINGS=1

free -m

module add cuda/10.0.130
module list

export CUDNN_HOME=/hpc/users/odonnt02/oss/cudnn/cuda
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_LIBRARY_PATH=$CUDNN_HOME/lib64:$CMAKE_LIBRARY_PATH
export INCLUDE_PATH=$CUDNN_HOME/include:$INCLUDE_PATH
export C_INCLUDE_PATH=$CUDNN_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:$CPLUS_INCLUDE_PATH
export CMAKE_INCLUDE_PATH=$CUDNN_HOME/include:$CMAKE_INCLUDE_PATH

python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'

env

cd {work_dir}

