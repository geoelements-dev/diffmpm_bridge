#!/bin/bash



module load intel/19.1.1
module load impi/19.0.9

ml cuda/12.2
ml cudnn
ml nccl

#module load mvapich2-gdr/2.3.7
#module load mvapich2/2.3.7ss

module load phdf5/1.10.4
module load python3/3.9.2

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH