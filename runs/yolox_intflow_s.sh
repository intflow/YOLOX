#!/bin/bash

python3 tools/train.py -f exps/intflow_total_100K/yolox_intflow_s.py -d 8 -b 64 --fp16 -o -c /data/pretrained/yolox_s.pth.tar
