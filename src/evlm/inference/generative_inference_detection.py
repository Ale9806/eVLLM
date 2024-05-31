import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from utils import save_output, add_alphabet, init_sub_results
import random


def evaluate_dataset(dataset: dict,
                     model_dict: dict[str, str],
                     split: str,
                     transform,
                     output_dir,
                     log_detection_imgs=False,
                     DEBUG: bool = True) -> None:
    """
    Evaluates a dataset using a given model.

    Parameters:
    dataset (dict): The dataset to be evaluated.
    model_dict (dict[str, str]): Dictionary containing model information.
    split (str): Split of the dataset (e.g., train, test).
    transform: Transformation function for the dataset.
    output_dir: Directory to save the output.
    DEBUG (bool): Whether to run in debug mode. Default is False.

    Returns:
    None
    """

    output_name = output_dir / model_dict['name'] / dataset["dataset"].name

    results = []
    for j, data_point in enumerate(
            tqdm(
                dataset["loader"],
                desc=
                f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}"
            )):
        data_point = add_class_name_prompt(data_point, dataset['dataset'].name)

        image_id: str = data_point["metadata"]['name']
        image: Path = dataset['dataset'].path / dataset[
            'dataset'].split / image_id
        sub_results: dict[str, str] = init_sub_results(data_point,
                                                       add_synonyms=True)
        # each element of `instances` has `className`s and `type`s. Each class has
        # 3 types: polygon, point, bbox.
        instances_polygon = data_point['instances'][0::3]
        instances_point = data_point['instances'][1::3]
        instances_bbox = data_point['instances'][2::3]
        if not all([ins['type'] == 'bbox' for ins in instances_bbox]):
            msg = f"{image_id} has badly formatted instace list"
            print(msg)
            # raise ValueError(msg)
            continue

        class_names = [ins['className'] for ins in instances_bbox]

        # for this image, one call per referring class
        for class_name in np.unique(class_names):
            result: dict = {}
            result['image_id'] = image_id
            result['class_name'] = class_name

            # gt bboxes for instances of this class
            gt_idxs = np.where(
                [ins['className'] == class_name for ins in instances_bbox])[0]
            if len(gt_idxs) == 0:
                continue
            gt_bboxes = [instances_bbox[idx]['points'] for idx in gt_idxs]
            result['gt_idxs'] = gt_idxs.tolist()
            result['gt_bboxes'] = gt_bboxes
            result['class_name_prompt'] = instances_bbox[
                gt_idxs[0]]['class_name_prompt']

            # referring object bbox
            output: dict = model_dict["model"].forward_detect(
                image, class_name=class_name)
            pred_bboxes = output['bboxes']
            result['pred_bboxes'] = pred_bboxes

            # some optional logging
            if log_detection_imgs and j < 20:
                do_bbox_logging(image,
                                gt_bboxes,
                                pred_bboxes,
                                dataset_name=dataset['dataset'].name,
                                class_name=class_name,
                                model_name=model_dict['name'],
                                j=j)

            results.append(result)

        # if j > 3:
        #     break

    f_out_stem = f"{str(output_name)}-detection.json"
    Path(f_out_stem).parent.mkdir(exist_ok=True, parents=True)
    with open(f_out_stem, 'w') as f:
        json.dump(results, f, indent=4)


def add_class_name_prompt(data_point, dataset_name):
    """
    choose an appropriate name for segmentation
    """

    # all classes are 'mitochondria' in wu et al
    if dataset_name == "wu_et_al_2023":
        for i in range(len(data_point['instances'])):
            data_point['instances'][i]['class_name_prompt'] = "mitochondria"

    # all classes are 'nucleus' in held et al
    elif dataset_name in ("held_et_al_2010_galt", "held_et_al_2010_h2b",
                          "held_et_al_2010_mt"):
        for i in range(len(data_point['instances'])):
            data_point['instances'][i]['class_name_prompt'] = "nucleus"

    # not a special case: just copy className
    else:
        for i in range(len(data_point['instances'])):
            data_point['instances'][i]['class_name_prompt'] = data_point[
                'instances'][i]['className']

    return data_point


def do_bbox_logging(image, gt_bboxes, pred_bboxes, dataset_name, class_name,
                    model_name, j):
    """ save an image of pred and gt bounding boxes """
    from PIL import Image, ImageDraw
    img = Image.open(image).convert("RGB")
    image_width, image_height = img.size
    draw = ImageDraw.Draw(img)
    for bbox in gt_bboxes:
        draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                       outline="red",
                       width=3)
    for bbox in pred_bboxes:
        draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                       outline="blue",
                       width=3)
    f_out = Path(f"image_seg/{dataset_name}/{model_name}/{class_name}_{j}.png")
    f_out.parent.mkdir(exist_ok=True, parents=True)
    img.save(f_out)
