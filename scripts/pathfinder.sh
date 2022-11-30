#!/bin/bash

relation=$1
dataset=$2
python sl_policy.py $relation $dataset
python policy_agent.py $relation retrain rl $dataset
python policy_agent.py $relation test rl $dataset

