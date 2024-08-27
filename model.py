import ast
import os
import pathlib

import neptune
import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# IMPORT NEPTUNE
project_name = f'mikmikthemimic/GM-Thesis3'
project = neptune.init_run(
    project=project_name,
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNmE3MzZjYS01MDUxLTQ2YjYtOGU3ZC04YzQwNjg4NmJiZDcifQ==",
    with_id="GMT3-488",
    mode="read-only",
)  # get project
experiment_id = "GMT3-488"  # experiment id
parameters = project['training/hyperparams'].fetch()

color_mapping = {
    1: "blue",
    2: "green",
    3: "white",
    4: "yellow",
    5: "red"
}

# load state dict
checkpoint = torch.load(params["MODEL_DIR"], map_location=device)

def get_faster_rcnn(
    backbone: torch.nn.Module,
    anchor_generator: AnchorGenerator,
    roi_pooler: MultiScaleRoIAlign,
    num_classes: int,
    image_mean: List[float] = [0.485, 0.456, 0.406],
    image_std: List[float] = [0.229, 0.224, 0.225],
    min_size: int = 512,
    max_size: int = 1024,
    **kwargs,
) -> FasterRCNN:
    model = FasterRCNN(
        backbone=backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        num_classes=num_classes,
        image_mean=image_mean,  # ImageNet
        image_std=image_std,  # ImageNet
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model

def Model() -> FasterRCNN:
    num_classes=6,
    anchor_size=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),),
    min_size=512,
    max_size=1024

    vgg_anchor_generator = AnchorGenerator(sizes=vgg_anchor_sizes, aspect_ratios=vgg_aspect_ratios)

    vgg_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    return get_faster_rcnn(
        backbone=backbone,
        anchor_generator=vgg_anchor_generator,
        roi_pooler=vgg_roi_pool,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )

if 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']

model_state_dict = {k.replace("model.", ""): v for k, v in checkpoint.items() if k.startswith("model.")}

#model.eval()
#model.to(device)
#for sample in dataloader_prediction:
#    x, x_name = sample
#    with torch.no_grad():
#        pred = model(x)
#        # Move tensors to CPU before converting to NumPy
#        pred = {key: value.cpu().numpy() for key, value in pred[0].items()}
#        name = pathlib.Path(x_name[0])
#        save_dir = pathlib.Path(os.getcwd()) / params["PREDICTIONS_PATH"]
#        save_dir.mkdir(parents=True, exist_ok=True)
#        pred_list = {
#            key: value.tolist() for key, value in pred.items()
#        }  # numpy arrays are not serializable -> .tolist()
#        save_json(pred_list, path=save_dir / name.with_suffix(".json"))