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
        questions:dict[str,str] = data_point['custom_metadata'][question_key]
        image_id:str = data_point["metadata"]['name']
        image:Path   = dataset['dataset'].path /image_id
        sub_results = {}
        
        meta_data:list[str] = ['microns_per_pixel',"domain","subdomain","modality","submodality","normal_or_abnormal"]
        for key in meta_data:
            sub_results[key] = data_point["custom_metadata"][key]

     

        for question_class in questions.keys():
            
            result:dict  = {}
            question = questions[question_class]["question"]
            answer   = questions[question_class]["answer"]
            options  = questions[question_class]["options"]
            
            assert answer in options 
            position = options.index(answer)

            
            if question_key == "questions":
                captions = [question + " " + option for option in options]

            elif question_key == "captions":
                captions = options

            #import pdb;pdb.set_trace()

            result["question_class"] = question_class
            result["questions"]      = captions
            result["image_id"]       = image_id
            result["correct_answer"] = answer
            result["correct_idx"]    = position
            output:dict         = model_dict["model"].forward(image,captions)
            result["model_answers"] = output
            

            if model_dict["name"] == "random":
                result = dict()
                model_answers = options.copy()
                random.shuffle(model_answers) 
                result["model_answers"] = model_answers

            
            for key in sub_results.keys():
                result[key] = sub_results[key]


            results.append(result)

        if DEBUG:
            if j == 2:
                import pdb;pdb.set_trace()
                break

    save_output(results, output_name)

