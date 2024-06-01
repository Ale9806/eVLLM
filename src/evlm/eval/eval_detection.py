"""
python src/evlm/eval/eval_detection.py

Process detection results processed by
	src/evlm/inference/generative_inference_detection.py
"""
from pathlib import Path
import sys
import ast
import pandas as pd
import os
import numpy as np
import json
from scipy.stats import bootstrap
from scipy.optimize import linear_sum_assignment

module_path = str(Path(__file__).resolve().parent)
sys.path.append(module_path)

sys.path.insert(0, "")
from src.evlm.inference.constants import DATASETS, ALL_MODELS


def process_model(model: str,
                  dataset: str,
                  round_to: int = 2,
                  filter_dict: dict[str, list] = None) -> dict[str]:
    """
	Summary stats for a single  experiment which is one model, one dataset
	The results have 
	"""
    if model not in ALL_MODELS:
        raise ValueError(f"model [{model}] not in model list {ALL_MODELS}")
    if dataset not in DATASETS:
        raise ValueError(f"DATASET [{dataset}] not in model list {DATASETS}")

    # load results
    f_preds = Path(f"outputs/{model}/{dataset}-detection.json")
    if not f_preds.exists():
        msg = f"No results file found [{f_preds}] for model [{model}] "
        msg += f"and dataset [{dataset}]. Need to run detection inference."
        raise ValueError(msg)
    with open(f_preds, 'r') as f:
        preds_dict = json.load(f)

    # filter
    if filter_dict is not None:
        raise NotImplementedErrror()

    # compute metrics per image
    for pred_image in preds_dict:
        gt_bboxes = [
            bbox_dict_to_list(bbox) for bbox in pred_image['gt_bboxes']
        ]
        pred_bboxes = [
            bbox_dict_to_list(bbox) for bbox in pred_image['pred_bboxes']
        ]
        pred_image['score_grit_loc'] = grit_localization_metric(
            pred_bboxes, gt_bboxes)

    # get per-class metrics
    class_names = [c['class_name'] for c in preds_dict]
    classes_uniq = np.unique(class_names).tolist()
    print(dataset)
    print(classes_uniq)

    results_dict = {}

    for class_name in classes_uniq:
        results_dict[class_name] = {}

        preds_this_class = [
            r for r in preds_dict if r['class_name'] == class_name
        ]

        class_name_prompt = preds_this_class[0]['class_name_prompt']
        score_grit_loc_mean = np.mean(
            [p['score_grit_loc'] for p in preds_this_class])

        results_dict[class_name]['class_name_prompt'] = class_name_prompt
        results_dict[class_name]['score_grit_loc_mean'] = score_grit_loc_mean

    return results_dict


def process_models(models: list[str],
                   datasets: list[str],
                   round_to: int = 2,
                   filter_dict: dict[str, list] = None) -> dict[str]:
    results_all_dict = {}
    results_all_lst = []

    for dataset in datasets:
        results_all_dict[dataset] = {}

        for model in models:
            result = process_model(model, dataset, round_to, filter_dict)
            results_all_dict[dataset][model] = result

            for class_, res in result.items():
            	results_all_lst.append([dataset, model, class_, res['class_name_prompt'], res['score_grit_loc_mean']])

    df = pd.DataFrame(results_all_lst, columns=['dataset','model','class','class_name_prompt', 'score_grit_loc'])

    return df, results_all_dict


def bbox_dict_to_list(bbox: dict) -> dict:
    """ 
	bbox from dict e.g. {'x1': 240.0, 'x2': 294.0, 'y1': 229.0, 'y2': 290.0}
	to a list expected by the functions below in order x1,y1,x2,y2
	"""
    return [bbox[c] for c in ['x1', 'y1', 'x2', 'y2']]


def compute_iou(bbox1: list, bbox2: list, verbose: bool = False):
    """
	src: https://github.com/allenai/grit_official/blob/main/metrics/localization.py
	"""
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
    area1 = compute_area(bbox1, invalid=0)
    area2 = compute_area(bbox2, invalid=0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou


def compute_area(bbox: list, invalid: float = None) -> float:
    """
	src: https://github.com/allenai/grit_official/blob/main/metrics/localization.py
	"""
    x1, y1, x2, y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1) * (y2 - y1)

    return area


def assign_boxes(pred_boxes: list[list], gt_boxes: list[list]):
    """
	src: https://github.com/allenai/grit_official/blob/main/metrics/localization.py
	"""
    n1 = len(pred_boxes)
    n2 = len(gt_boxes)
    cost = np.zeros([n1, n2])
    ious = np.zeros([n1, n2])
    for i, bbox1 in enumerate(pred_boxes):
        for j, bbox2 in enumerate(gt_boxes):
            iou = compute_iou(bbox1, bbox2)
            ious[i, j] = iou
            cost[i, j] = 1 - iou

    # solve assignment
    pred_box_ids, gt_box_ids = linear_sum_assignment(cost)
    pair_ids = list(zip(pred_box_ids, gt_box_ids))

    # select assignments with iou > 0
    pair_ids = [(i, j) for i, j in pair_ids if ious[i, j] > 0]
    pairs = [(pred_boxes[i], gt_boxes[j]) for i, j in pair_ids]
    pair_ious = [ious[i, j] for i, j in pair_ids]

    return pairs, pair_ious, pair_ids


def grit_localization_metric(pred_boxes: list[list],
                             gt_boxes: list[list]) -> float:
    """
	src: https://github.com/allenai/grit_official/blob/main/metrics/localization.py
	"""
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    if num_pred == 0 and num_gt == 0:
        return 1
    elif min(num_pred, num_gt) == 0 and max(num_pred, num_gt) > 0:
        return 0

    pairs, pair_ious, pair_ids = assign_boxes(pred_boxes, gt_boxes)
    num_detected = len(pairs)
    num_missed = num_gt - num_detected
    return np.sum(pair_ious) / (num_pred + num_missed)


if __name__ == "__main__":
    models = ["PaliGemma", "QwenVLM"]
    datasets = [
        "burgess_et_al_2024_contour", "burgess_et_al_2024_eccentricity",
        "burgess_et_al_2024_texture", "held_et_al_2010_galt",
        "held_et_al_2010_h2b", "held_et_al_2010_mt", "wu_et_al_2023"
    ]
    df, results_all = process_models(models, datasets)
    df.to_csv("summary_detection.csv")

    import pdb
    pdb.set_trace()
    pass
