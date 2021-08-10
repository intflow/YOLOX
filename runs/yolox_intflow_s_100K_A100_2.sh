#!/bin/bash

python3 tools/train.py -expn yolox_e2e_s-intflow_total_100K_2 -n yolox_e2e_intflow_s -f exps/intflow_total_100K_2/yolox_intflow_s.py -d 8 -b 64 --fp16 -o \
-c /data/pretrained/hcow/yolox_intflow_s-intflow_total_100K.pth.tar
