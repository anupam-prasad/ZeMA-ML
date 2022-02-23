#!/bin/bash
MAX_THREADS_NUMPY=${1:-4}  # use arg $1 or default 4

echo "Setting maximum number of threads for numpy to $MAX_THREADS_NUMPY"

export OMP_NUM_THREADS=$MAX_THREADS_NUMPY
export OPENBLAS_NUM_THREADS=$MAX_THREADS_NUMPY
export MKL_NUM_THREADS=$MAX_THREADS_NUMPY
export VECLIB_MAXIMUM_THREADS=$MAX_THREADS_NUMPY
export NUMEXPR_NUM_THREADS=$MAX_THREADS_NUMPY