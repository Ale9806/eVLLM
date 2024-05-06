import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils  import save_output
import random

def add_alphabet(options:list[str],answer:str):
        idx_to_option = {0:"A) ",1:"B) ",2:"C) ",3:"D) ",4:"E) ",5:"F) ",6:"G) ",7:"H) "}
        index = options.index(answer) 
        for i in range(0,len(options)):
            options[i] = idx_to_option[i] + options[i]

       
        
        answer =idx_to_option[index] + answer
        assert answer in options

        return options ,answer

def evaluate_dataset( dataset:dict, 
                       model_dict:dict[str,str], 
                       split:str,
                       transform,
                       output_dir,
                       DEBUG:bool ) -> None:

    output_name = output_dir / model_dict['name'] / dataset["data_name"] 
    results = []
    #import pdb;pdb.set_trace()

    # Inference
    for j, data_point in enumerate(tqdm( dataset["loader"], desc=f"Evaluating  {dataset['data_name']} | model:{model_dict['name']}")):
        

        
        questions:dict[str,str] = data_point['custom_metadata']["questions"]
        image_id:str = data_point["metadata"]['name']
        image:Path   = dataset["data_path"] /image_id
    
        for question in questions.keys():
            result:dict  = {}

            
            ##
            prompt:str         = "Answer the question, only providing the correspondig letter to the option, do not porvide more information:"
            question_str:str   = questions[question]["question"]
            answer:str         = questions[question]["answer"]
            options:list[str]  = questions[question]["options"]
            
            options ,answer = add_alphabet(options,answer)
            joined_options  =  '\n'.join(options)
            template:str   = prompt +"\n" + question_str + "\n" + joined_options

            ##
            #import pdb;pdb.set_trace()
            result["question_class"] = question
            result["questions"]      = template
            result["image_id"]       = image_id
            
            result["correct_answer"] = answer
            output:dict         = model_dict["model"].forward(image,template)
            result["model_answers"] = output

            if model_dict["name"] == "random":
                result = dict()
                model_answers = options.copy()
                random.shuffle(model_answers) 
                result["model_answers"] = model_answers


            results.append(result)

        if DEBUG:
            if j == 2:
                import pdb;pdb.set_trace()
                break

    save_output(results, output_name)

