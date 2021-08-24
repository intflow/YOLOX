# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.depth = 1.33
        self.width = 1.25
# ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.random_size = (14, 26)
        self.train_path = '/data/EdgeFarm_cow/intflow_total_100K_2'
        self.val_path = '/data/EdgeFarm_cow/intflow_total_1K'
        self.train_ann = "label_odtk_025pi_center.json"
        self.val_ann = "label_coco_rbbox.json"

        # --------------- transform config ----------------- #
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = (0.5, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 100
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.5
        self.nmsthre = 0.65

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data.datasets.intflow import INTFLOWDataset
        from yolox.data.datasets.mosaicdetection import MosaicDetection
        from yolox.data.data_augment import TrainTransform
        from yolox.data.dataloading import DataLoader
        from yolox.data.samplers import InfiniteSampler, YoloBatchSampler
        import torch.distributed as dist

        dataset = INTFLOWDataset(
                data_dir=self.train_path,
                json_file=self.train_ann,
                name="img_mask",
                img_size=self.input_size,
                preproc=TrainTransform(max_labels=50),
                rotation=True,
                compatible_coco=False,
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        else:
            sampler = torch.utils.data.RandomSampler(self.dataset)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import INTFLOWDataset, ValTransform

        valdataset = INTFLOWDataset(
            data_dir=self.val_path,
            json_file=self.val_ann,
            name="img_mask",
            img_size=self.input_size,
            preproc=ValTransform(),
            compatible_coco=True,
            rotation=False,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import INTFLOWEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = INTFLOWEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
