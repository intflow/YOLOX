#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import sys
import pandas as pd
import re
import shutil

import yaml

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, INTFLOW_CLASSES, CROWDHUMAN_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import yolox.utils.boxes as B

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment_name", type=str, default="yolox_s_oad_lm3__intflow_total_1K")
    parser.add_argument("-n", "--name", type=str, default="yolox_s_oad_lm3", help="model name")

    parser.add_argument(
        "--path", default="./assets/cow1.jpg", help="path to images or video"
    )
    parser.add_argument(
        "--save_folder", default=None, help="path to images or video output"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/yolox_oad_lm3__intflow_total_100K_2/yolox_x_oad_lm3.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/data/pretrained/hcow/yolox_x_oad_lm3__intflow_total_100K_2_test2.pth", type=str, help="ckpt for eval")
    #parser.add_argument("-m", "--model", default=None, type=str, help="model reference for eval")
    #parser.add_argument("-c", "--ckpt", default="/data/pretrained/hcow/yolox_s_oad_lm3__intflow_total_1K_p0.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.6, type=float, help="test conf")
    parser.add_argument("--nms", default=0.0005, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--pruning",
        dest="pruning",
        default=False,
        action="store_true",
        help="Set pretrained model is whether pruned or not",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        landmarks = output[:, 8:8+2*3]

        # preprocessing: resize
        bboxes /= ratio
        landmarks /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        scores = torch.sqrt(scores + 1e-6)
        rads = output[:,7]

        vis_res = vis(img, bboxes, rads, scores, cls, landmarks, cls_conf, self.cls_names)

        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result, save_folder=None):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            if save_folder == None:
                save_folder = os.path.join(
                    vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args, save_folder=None):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_folder == None:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

# ANCHOR : pull data
def pull_data(predictor, vis_folder, current_time, args, save_folder=None):

    cap = cv2.VideoCapture(args.path if args.demo == "video" or args.demo == "pull" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        if save_folder == None:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video" or args.demo == "pull":
            save_path = os.path.join(save_folder, args.path.split("/")[-1])
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
    
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    df = None
    frame_ = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val:

            outputs, img_info = predictor.inference(frame)
            if outputs[0] is not None:
                np_data = outputs[0].cpu().numpy()
                np_data[:,0:4] = B.xyxy2cxcywh(np_data[:,0:4] / img_info['ratio'])
                np_data[:,8:] = np_data[:,8:] / img_info['ratio']

                if df is None:
                    df = pd.DataFrame(np_data)
                    df['frame'] = frame_
                else:
                    df_p = pd.DataFrame(np_data)
                    df_p['frame'] = frame_
                    df = pd.concat([df,df_p])

            frame_ += 1

            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            df.columns = ["xc","yc","width","height","score1","score2","class","theta","no_x","no_y","ne_x","ne_y","ta_x","ta_y","frame"]
            df["class"]  = df["class"].astype("int")
            df = df[["frame","xc","yc","width","height","theta","no_x","no_y","ne_x","ne_y","ta_x","ta_y"]]
            # df.to_csv("test.csv", sep=",", index=False)
            f_name = os.path.splitext(os.path.basename(args.path))[0]

            if not os.path.isdir(os.path.join(args.sv_path, f_name)):
                os.mkdir(os.path.join(args.sv_path, f_name))
            df_sv_name = os.path.join(args.sv_path, f_name, f_name)+"_det.txt"
            df.to_csv(df_sv_name, sep=",", index=False)

            if args.ori_video_cp:
                dst = os.path.join(args.sv_path, f_name, os.path.basename(args.path))
                shutil.copy(args.path, dst)

            #* making .ini file -----------------------------------------------------------------------------------
            config_path = os.path.join(os.path.dirname(__file__), "config.ini")
            f = open(config_path, "w")
            line1 = 'det_file_path="{}"\n'.format(df_sv_name)
            if args.ori_video_cp:
                line2 = 'video_file_path="{}"\n'.format(dst)
            else:
                line2 = 'video_file_path="{}"\n'.format(args.path)
            line3 = 'out_trk_path="{}"\n'.format(re.sub("_det", "_trk",df_sv_name))
            line4 = 'video_output_path="{}"\n'.format(re.sub("txt","avi",re.sub("_det", "_trk",df_sv_name)))
            line5 = 'is_column=true\n'
            line6 = 'video_write=true\n'
            line7 = 'color_number=150\n'
            line8 = 's_video_width={}\n'.format(img_info['width'])
            line9 = 's_video_height={}'.format(img_info['height'])
            for i in range(1,10):
                f.write(eval("line"+str(i)))
            f.close()
            #* ----------------------------------------------------------------------------------------------------
            #? making .yaml file -----------------------------------------------------------------------------------
            yaml_dict = {}
            if args.ori_video_cp:
                yaml_dict["video_file_path"] = dst
            else:
                yaml_dict["video_file_path"] = args.path
            yaml_dict["trk_file_path"] = re.sub("_det", "_trk",df_sv_name)
            yaml_dict["output_path"] = args.sv_path
            yaml_dict["column"] = False
            yaml_dict["width_roi_padding"] = 0.0
            yaml_dict["height_roi_padding"] = 0.0
            yaml_dict["masking"] = True
            yaml_dict["width"] = 128
            yaml_dict["height"] = 128
            
            with open(os.path.join(os.path.dirname(__file__), "cutter_f1_config.yaml"), 'w', encoding="utf-8") as outfile:
	            yaml.dump(yaml_dict, outfile, default_flow_style=False, allow_unicode=True)
            #? ----------------------------------------------------------------------------------------------------

            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)


    if args.pruning:
        model = exp.get_model_pruning(args.model)
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    else:
        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    classes_list = INTFLOW_CLASSES
    predictor = Predictor(model, exp, classes_list, trt_file, decoder, args.device, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result, args.save_folder)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args, args.save_folder)
    elif args.demo == "pull":
        pull_data(predictor, vis_folder, current_time, args, args.save_folder)
    

def get_yaml(path):

    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":
    args = make_parser().parse_args()
    config = get_yaml("/works/YOLOX/tools/self_label.yaml")
    #* -----------------------------------------
    args.__dict__ = config
    #* -----------------------------------------
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
