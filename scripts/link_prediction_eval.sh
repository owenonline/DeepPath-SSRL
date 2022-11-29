#!/bin/bash

relation=$1
dataset=$2
python evaluate.py $relation $dataset
python transR_eval.py $relation $dataset
python transE_eval.py $relation $dataset