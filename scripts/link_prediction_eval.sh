#!/bin/bash

relation=$1
dataset=$2
method=$3
python evaluate.py $relation $dataset $method
python transR_eval.py $relation $dataset $method
python transE_eval.py $relation $dataset $method