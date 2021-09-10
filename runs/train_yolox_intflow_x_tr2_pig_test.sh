#!/bin/bash

python3 tools/train.py -expn yolox_x_oad_lm3__tr2_pig -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__tr2_pig/yolox_x_oad_lm3_test.py -d 4 -b 32 --fp16 \
-c /data/pretrained/yolox_x.pth
