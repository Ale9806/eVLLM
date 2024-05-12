import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils  import save_output
import random

def evaluate_dataset( dataset:dict, 
                       model_dict:dict[str,str], 
                       split:str,
                       transform,
                       output_dir,
                       question_key:str = "captions",
                       DEBUG:bool=False) -> None:


    output_name = output_dir / model_dict['name'] / dataset["dataset"].name
    results = []
  
    for j, data_point in enumerate(tqdm( dataset["loader"], desc=f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}")):
        images,classes = [],[]        
        for image,classes_ in data_point:
            images.append(image)
            classes.append(classes_)
            
        output:dict = model_dict["model"].forward(images,dataset["dataset"].class_names)
       

        for idx in range(0,len(output["probs"])):
            result = {}
            result["image_id"]       = images[idx]
            result["correct_answer"] = classes[idx]
            result["probs"]          = output["probs"][idx]
            results.append(result)
            #
    save_output(results, output_name)

