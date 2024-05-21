#!/bin/bash
# to execute:
# chmod +x source.sh
# ./source.sh

set -e # exit if nonzero status

CONDA_PATH="/Users/sebastiandixon/opt/anaconda3"
VENV_CIAO="ciao"
VENV_BASE="base"

SCRIPT_BASE="source_base.py"
SCRIPT_CIAO="source_ciao.py"

ROUND=1
NUM_SIMS=1000
NUM_ROUNDS=10


run_in_ciao() {
    source $CONDA_PATH/bin/activate $VENV_CIAO
    export ROUND
    export NUM_SIMS
    echo "Running $1 in ciao with ROUND=$ROUND"
    python $1
    conda deactivate
}


run_in_base() {
    source $CONDA_PATH/bin/activate $VENV_BASE
    export ROUND
    export NUM_SIMS
    echo "Running $1 in base"
    python $1
    conda deactivate
}


for ((i=1; i<=NUM_ROUNDS; i++))
do
    run_in_base $SCRIPT_BASE
    ROUND=$((ROUND + 1))
    run_in_ciao $SCRIPT_CIAO
    
done