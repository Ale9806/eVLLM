import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils  import save_output,add_alphabet,init_sub_results
import random

def evaluate_dataset( dataset:dict, 
                       model_dict:dict[str,str], 
                       split:str,
                       transform,
                       output_dir,
                       prompt:str = "Answer with a single letter, no extra details.",
                       question_key:str="questions",
                       DEBUG:bool=True) -> None:
    """
    Evaluates a dataset using a given model.

    Parameters:
    dataset (dict): The dataset to be evaluated.
    model_dict (dict[str, str]): Dictionary containing model information.
    split (str): Split of the dataset (e.g., train, test).
    transform: Transformation function for the dataset.
    output_dir: Directory to save the output.
    prompt (str): Prompt for answering questions. Default is "Answer with a single letter, no extra details."
    DEBUG (bool): Whether to run in debug mode. Default is False.

    Returns:
    None
    """
    output_name = output_dir / model_dict['name'] / dataset["dataset"].name
    results = []
    for j, data_point in enumerate(tqdm( dataset["loader"], desc=f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}")):
        questions:dict[str,str] = data_point['custom_metadata'][question_key]
        image_id:str = data_point["metadata"]['name']
        image:Path = dataset['dataset'].path / dataset['dataset'].split / image_id
        sub_results:dict[str,str] = init_sub_results(data_point,add_synonyms=True)
    
        for question in questions.keys():
            result:dict  = {}
            question_str:str   = questions[question]["question"]
            answer:str         = questions[question]["answer"]
            options:list[str]  = questions[question]["options"]
            position:int = options.index(answer)
            
            options ,answer = add_alphabet(options,answer)
            joined_options:str =  '\n'.join(options)
            template:str = prompt +"\n" + "Question:" + question_str + "\n" + joined_options

            output:dict         = model_dict["model"].forward(image,template)
            result["question_class"] = question
            result["questions"]      = template
            result["image_id"]       = image_id
            result["correct_answer"] = answer
            result["correct_idx"]    = position
            result["model_answers"]  = output

            for key in sub_results.keys():
                result[key] = sub_results[key]

            results.append(result)

        if DEBUG:
            if j == 2:
                import pdb;pdb.set_trace()
                break
                
    save_output(results, output_name)

