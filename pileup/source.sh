#!/bin/bash

VENV_CIAO="ciao/bin/activate"
VENV_BASE="base/bin/activate"

SCRIPT_BASE="source_base.py"
SCRIPT_CIAO="source_ciao.py"

ROUND=1

run_in_ciao() {
    source $VENV_CIAO
    export ROUND
    echo "Running $1 in ciao with ROUND=$ROUND"
    python $1
    deactivate
    ROUND=$((ROUND + 1))
}

run_in_base() {
    source $VENV_BASE
    unset ROUND
    echo "Running $1 in base"
    python $1
    deactivate
}

run_in_ciao $SCRIPT_BASE
run_in_base $SCRIPT_CIAO
run_in_ciao $SCRIPT_BASE
run_in_base $SCRIPT_CIAO