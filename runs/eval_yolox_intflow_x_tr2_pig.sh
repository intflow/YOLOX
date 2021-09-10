#!/bin/bash

python3 tools/eval.py -expn yolox_x_oad_lm3__tr2_pig -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__tr2_pig/yolox_x_oad_lm3_test.py -d 4 -b 32 --fp16 \
-c YOLOX_outputs/yolox_x_oad_lm3__tr2_pig/best_ckpt.pth \
--conf 0.001 --nms 0.45
