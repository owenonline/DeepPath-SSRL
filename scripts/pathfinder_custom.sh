#!/bin/bash

relation=$1
dataset=$2
python sl_policy_custom.py $relation $dataset
python policy_agent.py $relation retrain ssrl $dataset
python policy_agent.py $relation test ssrl $dataset