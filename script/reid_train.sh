#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

set -x

cd fast-reid
python3 extract_crop.py
python3 tools/train_net.py --config-file ./configs/AIC24/mgn_R101_reprod.yml --num-gpus 4
python3 tools/convert_weight.py