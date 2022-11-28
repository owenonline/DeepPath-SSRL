#!/bin/bash

relation=$1
python sl_policy_custom.py $relation
python policy_agent.py $relation retrain
python policy_agent.py $relation test