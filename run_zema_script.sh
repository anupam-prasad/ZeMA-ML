#!/bin/bash

SCALING=${1} # use arg $1 for scaling. Default 1, i.e scale dataset 
. ~/zema-ml/venv/bin/activate
python ~/zema-ml/zema_generate_model.py $SCALING
