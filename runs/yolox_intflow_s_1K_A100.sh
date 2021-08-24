#!/bin/bash

python3 tools/train.py -expn yolox_s_oad_lm3__intflow_total_1K -n yolox_s_oad_lm3 -f exps/yolox_oad_lm3__intflow_total_1K/yolox_s_oad_lm3.py -d 8 -b 64 --cache --fp16 \
-c /data/pretrained/yolox_s.pth
