#!/bin/bash

dataset=$1
custom=$2
method=$3
for d in ./$dataset/tasks/*/; do
    [ -L "${d%/}" ] && continue
    echo "$d"
    ./link_prediction_eval.sh $d $dataset $method
done