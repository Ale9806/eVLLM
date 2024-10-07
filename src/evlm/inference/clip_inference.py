import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from utils  import save_output,init_sub_results
import random


#QUESTIONS = ['modality', 'submodality', 'domain', 'subdomain' , 'stain', 'classification']
def evaluate_dataset( dataset:dict, 
                       model_dict:dict[str,str], 
                       split:str,
                       transform,
                       output_dir,
                       question_key:str = "captions",
                       DEBUG:bool=False) -> None:
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
    for j, data_point in enumerate(tqdm( dataset["loader"], desc=f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}")):
        if dataset['dataset'].name == "cognition":
            print("setting question key to questions")
            question_key:str = "questions"
        else:
            pass
        print(f"Doing infernece with {question_key}")
        questions:dict[str,str] = data_point['custom_metadata'][question_key]
        image_id:str = data_point["metadata"]['name']
        if dataset['dataset'].name == "cognition":
            image:Path = dataset['dataset'].path / image_id
        else:
            image:Path = dataset['dataset'].path / dataset['dataset'].split / image_id
        sub_results:dict[str,str] = init_sub_results(data_point)
        
          
        for question_class in questions.keys():
            #import pdb;pdb.set_trace()
            result:dict  = {}
            question:str = questions[question_class]["question"]
            answer:str   = questions[question_class]["answer"]
            options:list[str] = questions[question_class]["options"]
            position:int = options.index(answer)

            
        
            assert answer in options,f"answer not in options: {answer} not in {options}"
            
            if question_key == "questions":
                options = [question + " " + option for option in options]

            #import pdb; pdb.set_trace()
            try:
                output:dict = model_dict["model"].forward(image,options)
            except Exception as e:
                    print(f"Could not run inference for {image_id}, error: {e}")

            #import pdb;pdb.set_trace()

            result["question_class"] = question_class
            result["questions"]      = options
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


#image:Path   = dataset['dataset'].path /image_id