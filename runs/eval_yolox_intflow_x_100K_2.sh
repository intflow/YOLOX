#!/bin/bash

python3 tools/eval.py -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py -d 4 -b 32 --fp16 \
-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test3.pth \
--conf 0.5 --nms 0.45
