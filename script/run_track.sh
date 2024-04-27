#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

set -x

for i in $(ls result/detection); do
OMP_NUM_THREADS=1 python track/run_tracking_batch.py $i  > result/track_log1/$i.txt&
done
#
wait
