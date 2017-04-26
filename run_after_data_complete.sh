#!/usr/bin/env bash

python3 valiation_functions.py

python3 regressors/testing_loops.py -tpot -g 5 -p 10 -n 1

lrztar tpot_results

./deploy_data.sh

