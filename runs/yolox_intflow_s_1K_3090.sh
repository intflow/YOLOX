#!/bin/bash

python3 tools/train.py -expn intflow_total_1K -n yolox_e2e_intflow_s -f exps/intflow_total_1K/yolox_intflow_s.py -d 4 -b 64 --fp16 -o -c /data/pretrained/yolox_s.pth.tar
