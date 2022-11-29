#!/bin/bash

dataset=$1
custom=$2
for d in ./$dataset/tasks/*/; do
    [ -L "${d%/}" ] && continue
    echo "$d"
    if [ $custom == 0 ]
    then
        ./pathfinder.sh $d $dataset
    else
        ./pathfinder_custom.sh $d $dataset
    fi
    ./link_prediction_eval.sh $d $dataset
done