#!/bin/bash

python3 tools/train.py -expn yolox_s_oad_lm3__intflow_total_100K_2 -n yolox_s_oad_lm3 -f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_s_oad_lm3.py -d 4 -b 32 --cache --fp16 \
-c /data/pretrained/yolox_s.pth
