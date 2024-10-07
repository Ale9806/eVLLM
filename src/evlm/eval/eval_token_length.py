from pathlib import Path
import sys
import ast
import pandas as pd
import os
import numpy as np

module_path = str(Path(__file__).resolve().parent) 
sys.path.append(module_path)

from eval_utils import *
from eval_utils import tasks_metadata
from plot_utils import *


def accuracy_confidence_regression(
    models: list[str], 
    datasets: list[str], 
    round_to: int = 2, 
    extension: str = ".csv",
    filter_dict:dict[str,list]=None) -> list[dict[str]]:
    """
    Used to identify instance indentification
    Process data from multiple models and datasets and calculate evaluation metrics.

    Parameters:
        models (List[str]): A list of model names.
        datasets (List[str]): A list of dataset names.
        round_to (int, optional): Number of decimal places to round the results to. Default is 3.
        extension (str, optional): The file extension of the data files. Default is ".csv".

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each model.
    """
    results = []
    dfs = []
    for model in models:
        print(f"Trying:{model}")
        for dataset in datasets:
            append = True
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
            
                ### Filter ###
                if filter_dict:
                    sub_df, append = filter_dataset(sub_df,filter_dict)
                
                if append:
                    sub_df["model"]    = model
                    sub_df["dataset"] = dataset
                    dfs.append(sub_df)

            else:
                print(f"No results for {dataset}")

        #import pdb;pdb.set_trace()
        
        df = pd.concat(dfs, ignore_index=True)
        df["char_length"] = df["correct_answer"].apply(len)
        replace_nan(df,with_zero=True)
        df['confidence']  = df.apply(get_confidence, axis=1)
        get_results(df, model)

        
        
        #import pdb;pdb.set_trace()


        #df["question_class"]  == classification_0
        #df["dataset"]  =  df["question_class"] 
        df.loc[df["question_class"] == "classification_0", "question_class"] = df.loc[df["question_class"] == "classification_0", "dataset"]
        plot_calibration(df,f"{model}_confidence_correctnes_correlation")
        plot_length_confidence(df,f"{model}_correct_inccorect")
       
        #accuracy_df = df.groupby('question_class')[["is_correct",'confidence']].mean().round(round_to).reset_index()
       
    #import pdb;pdb.set_trace()

    return results



if __name__ == "__main__":


    round_to           = 2
    models:list[str]   = ["OpenCLIP_lion","OpenCLIP","QuiltCLIPRobust","PLIPRobust","CLIP","ALIGN","QuiltCLIP","PLIP","BioMedCLIP","ConchCLIP","QwenVLM","CogVLM","QwenVLM"] #"PaliGemma"
    datasets:list[str] = tasks_metadata.keys()
    extension:str = ".csv"
    filter_dict:dict[str,list[str]] = {"modality":["light microscopy"],"domain":["pathology"]}
    filter_dict:dict[str,list[str]] = None
    filenmae:str = "eval"
    if filter_dict:
        filenmae =construct_filter_name(filenmae,filter_dict) 
  
    output_path:str    = pathlib.Path("outputs","tables",filenmae )

    results = accuracy_confidence_regression(models=models, 
                             datasets=datasets,
                             filter_dict = filter_dict)

   
    import pdb; pdb.set_trace()



