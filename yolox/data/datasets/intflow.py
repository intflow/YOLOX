#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Intflow, Inc. and its affiliates.

import cv2
import numpy as np
from pycocotools.coco import COCO
from loguru import logger

import os

from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.datasets_wrapper import Dataset
import yolox.utils.boxes as B
import yolox.utils.visualize as V
from PIL import Image
import torch


class INTFLOWDataset(Dataset):
    """
    INTFLOW dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="label_odtk_025pi_center.json",
        name="img_mask",
        img_size=(416, 416),
        preproc=None,
        rotation=True,
        compatible_coco=False,
        cache=False,
    ):
        """
        INTFLOW dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): INTFLOW json file name
            name (str): INTFLOW data name (e.g. 'img' or 'img_mask')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "INTFLOW")
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.rotation = rotation
        self.compatible_coco = compatible_coco
        self.coco = COCO(os.path.join(self.data_dir, self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_intflow_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_intflow_annotations(self):
        if self.rotation == True:
            annots = [self.load_anno_from_ids_rbbox(_ids) for _ids in self.ids]
        elif self.compatible_coco == True:
            annots = [self.load_anno_from_ids_coco(_ids) for _ids in self.ids]
        else:
            annots = [self.load_anno_from_ids_bbox(_ids) for _ids in self.ids]

        return annots

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the frist time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids_rbbox(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1,y1,x2,y2,rad = B.rotated_reform(obj["bbox"][0:5], width, height)
            if obj["area"] > 0 and x2 > x1 and y2 > y1:
                obj["clean_bbox"] = [x1,y1,x2,y2,rad]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6+2*3))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:5] = obj["clean_bbox"]
            res[ix, 5] = cls
            res[ix, 6:6+2*3] = obj["landmark"]

        img_info = (height, width)

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"

        del im_ann, annotations

        ###### Write overlay image for debugging
        ####img_name = self.coco.loadImgs(id_)[0]['file_name']
        ####im = Image.open('{}/{}/{}'.format(self.data_dir, self.name, img_name)).convert("RGB")
        ####data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        ####data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        ####V.write_overlay(data, res, id_, 'tmp_figs') #Use only for visual debug on image augmentation an its label

        return (res, img_info, file_name)

    def load_anno_from_ids_bbox(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:  
            x1,y1,x2,y2 = B.rotated2rect(obj["bbox"][0:5], width, height)
            if obj["area"] > 0 and x2 > x1 and y2 > y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        img_info = (height, width)

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"

        del im_ann, annotations

        ##### Write overlay image for debugging
        ###img_name = self.coco.loadImgs(id_)[0]['file_name']
        ###im = Image.open('{}/{}/{}'.format(self.data_dir, self.name, img_name)).convert("RGB")
        ###data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        ###data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        ###V.write_overlay(data, res, id_, 'tmp_figs') #Use only for visual debug on image augmentation an its label

        return (res, img_info, file_name)        

    def load_anno_from_ids_coco(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 > x1 and y2 > y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        img_info = (height, width)

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"

        del im_ann, annotations

        ###### Write overlay image for debugging
        ####img_name = self.coco.loadImgs(id_)[0]['file_name']
        ####im = Image.open('{}/{}/{}'.format(self.data_dir, self.name, img_name)).convert("RGB")
        ####data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        ####data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        ####V.write_overlay(data, res, id_, 'tmp_figs') #Use only for visual debug on image augmentation an its label

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )

        img = cv2.imread(img_file)
        assert img is not None

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
