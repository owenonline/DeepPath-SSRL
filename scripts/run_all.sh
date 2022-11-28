#!/bin/bash

dataset = $1
custom = $2
for d in $dataset/*/ ; do
    [ -L "${d%/}" ] && continue
    if [ $custom == 0 ]
    then
        ./pathfinder.sh $d
    else
        ./pathfinder_custom.sh $d
    fi
    ./link_prediction_eval.sh $d
done