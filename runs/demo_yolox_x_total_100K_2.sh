##python3 tools/demo.py --demo video -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
##-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
##-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
##--path /data/EdgeFarm_pig/video_sample/AIDKR_202109020409.mp4 \
##--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

##python3 tools/demo.py --demo video -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
##-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
##-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
##--path /data/EdgeFarm_pig/video_sample/202109040240.mp4 \
##--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

##python3 tools/demo.py --demo video -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
##-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
##-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
##--path /data/EdgeFarm_pig/video_sample/202109040256.mp4 \
##--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

##python3 tools/demo.py --demo image -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
##-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
##-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
##--path ./assets/pig.jpg \
##--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

##python3 tools/demo.py --demo video -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
##-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
##-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
##--path /data/EdgeFarm_pig/video_sample/202109040256.mp4 \
##--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

python3 tools/demo.py --demo video -expn yolox_x_oad_lm3__intflow_total_100K_2 -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py \
-c /data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth \
--path /data/EdgeFarm_cow/video_sample/moonfarm_densecow_20210720_160508447.mp4 \
--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

