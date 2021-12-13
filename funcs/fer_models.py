"""
This file contains Facial Expression Recognition method.
"""

import cv2
import torch

from funcs.fer_model_vgg16 import vgg16face

model_list = ["LeNet", "vgg16", "ViT"]


def createFERmodel(model_name, weights_dir="random", cuda=True):
    model = None
    if model_name == "vgg16":
        model = vgg16face(weights_dir=weights_dir, cuda=cuda)

    elif model_name == "ViT":
        pass

    else:
        raise Exception("Unknown FER model.")

    return model
