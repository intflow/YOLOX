#!/bin/bash

python3 tools/train.py -expn intflow_total_100K -n yolox_intflow_s -f exps/intflow_total_100K/yolox_intflow_s.py -d 8 -b 128 --fp16 -o -c /data/pretrained/yolox_s.pth.tar
