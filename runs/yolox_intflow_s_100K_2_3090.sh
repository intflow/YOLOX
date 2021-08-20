#!/bin/bash

python3 tools/train.py -expn yolox_oad_e2e_s-intflow_total_100K_2 -n yolox_oad_e2e_s -f exps/intflow_oad__total_100K_2/yolox_intflow_s.py -d 4 -b 128 --fp16 -o \
-c /data/pretrained/hcow/yolox_intflow_s-intflow_total_100K.pth.tar