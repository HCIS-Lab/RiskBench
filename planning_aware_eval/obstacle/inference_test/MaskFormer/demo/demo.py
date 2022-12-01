# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from inference_test.MaskFormer.mask_former import add_mask_former_config
from inference_test.MaskFormer.demo.predictor import VisualizationDemo


# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg(device):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file("./inference_test/MaskFormer/configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml")
    # cfg.merge_from_list(args.opts)
    # cfg.MODEL.WEIGHTS = 'model_final_f3fc73.pkl'
    # cfg.freeze()
    if device == "cpu":
        cfg.MODEL.DEVICE = device
    return cfg

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def get_maskformer(device,mapillary_pretrained=True):
    cfg = setup_cfg(device)
    
    import gdown
    if not os.path.exists('./inference_test/MaskFormer/demo/'):
        os.mkdir('./inference_test/MaskFormer/demo/')

    if not os.path.isfile('./inference_test/MaskFormer/demo/model_final_f3fc73.pkl'):
        print("Download maskformer weight")
        url = "https://drive.google.com/u/4/uc?id=18ZeXFEJWFehoQNvryhDqKlWf0yNxok02&export=download"
        gdown.download(url, './inference_test/MaskFormer/demo/model_final_f3fc73.pkl')


    if mapillary_pretrained:
        # cfg.MODEL.WEIGHTS = '/data/hanku/Interaction-benchmark/models/MaskFormer/demo/model_final_f3fc73.pkl'
        cfg.MODEL.WEIGHTS = './inference_test/MaskFormer/demo/model_final_f3fc73.pkl'
    else:
        cfg.MODEL.WEIGHTS = ''
    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    return model
