#!/bin/bash

directory=$1

cd $directory
python metrics.py
export PYTHONUNBUFFERED=1
python compute.py