#!/bin/bash

relation=$1
dataset=$2
python sl_policy.py $relation
python policy_agent.py $relation retrain rl
python policy_agent.py $relation test rl

