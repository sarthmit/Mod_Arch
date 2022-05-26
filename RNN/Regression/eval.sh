#!/bin/bash

source activate pytorch

export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

for gt in 2 4 8 16 32; do
  python perf.py "$@" --gt-rules $gt --num-rules $gt
  python spec.py "$@" --gt-rules $gt --num-rules $gt
  python prob.py "$@" --gt-rules $gt --num-rules $gt

  python perf.py "$@" --gt-rules $gt --num-rules $gt --best
  python spec.py "$@" --gt-rules $gt --num-rules $gt --best
  python prob.py "$@" --gt-rules $gt --num-rules $gt --best
done

echo Job is over