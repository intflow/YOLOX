python3 tools/demo.py --demo image -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
--path ./assets/pig2.jpg \
--conf 0.6 --nms 0.65 --tsize 640 --save_result --device gpu