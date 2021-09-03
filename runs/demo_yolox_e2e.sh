python3 tools/demo.py --demo video -expn yolox_s_oad_lm3__intflow_total_100K_2 -n yolox_s_oad_lm3 \
-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_s_oad_lm3.py \
-c /data/pretrained/hcow/yolox_s_oad_lm3__intflow_total_100K_2_test1.pth \
--path /data/EdgeFarm_pig/video_sample/AIDKR_202109020409.mp4 \
--conf 0.6 --nms 0.65 --tsize 640 --save_result --device gpu