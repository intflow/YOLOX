#!/bin/bash

python3 tools/train.py -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py -d 8 -b 64 --resume --fp16 \
-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth
