#!/bin/bash

module reset 

# Load modules
module load python3/3.9
module load cuda/12.2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/intel19/python3/3.9.2/lib/

# create env
# ---------
python3 -m virtualenv venv
source venv/bin/activate

python -m pip install --upgrade pip
# install torch with CUDA 12.1 support (which matches cuda/12.2)
pip install taichi==1.5

pip install -r requirements.txt

# test env
# --------

# Check if the first command line argument is "--run-tests=true"
if [ "$1" = "--run-tests=true" ]; then
  echo 'Running tests...'
  
  echo 'which python -> venv'
  which python
  
  echo 'test_pytorch.py -> random tensor'
  python3 test/test_pytorch.py
  
  echo 'test_pytorch_cuda_gpu.py -> True if GPU'
  python3 test/test_pytorch_cuda_gpu.py
  
  echo 'test_torch_geometric.py -> no return if import successful'
  python3 test/test_torch_geometric.py
  
else
  echo "Skipping tests. To run tests, use the argument --run-tests=true"
fi

# Clean up
# --------
#rm -r venv