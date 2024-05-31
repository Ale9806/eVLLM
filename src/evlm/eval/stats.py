from pathlib import Path
import sys
import ast
import pandas as pd
import os
import numpy as np
from scipy.stats import bootstrap

module_path = str(Path(__file__).resolve().parent) 
sys.path.append(module_path)

from eval_utils import *

   



def process_models_question_only(
    models: list[str], 
    datasets: list[str], 
    round_to: int = 2, 
    extension: str = ".csv",
    filter_dict:dict[str,list]=None,
    tasks_metadata:dict[str,dict]=None,
    to_percentage:bool=False,
    question_name:str=["classification_0","classification_1"]) -> list[dict[str]]:
    
    """
    Used to evaluate instance classifcation
    Process data from multiple models and datasets and calculate evaluation metrics.

    Parameters:
        models (List[str]): A list of model names.
        datasets (List[str]): A list of dataset names.
        round_to (int, optional): Number of decimal places to round the results to. Default is 3.
        extension (str, optional): The file extension of the data files. Default is ".csv".

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each model.
    """
    if to_percentage:
        multiplier = 100
    else:
        multiplier = 1

    results = []
    dfs = []
    for model in models:
        result = {}
        for dataset in datasets:
            append = True
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
            
                ### Filter ###
                if filter_dict:
                    sub_df, append = filter_dataset(sub_df,filter_dict)
                
                if append:
                    #print(f"added {dataset}")
                    sub_df,_ = filter_dataset(sub_df,{"question_class":question_name})
                    sub_df["dataset"] = dataset
                    sub_df["model"]   = model
                    #import pdb;pdb.set_trace()
                    get_results(sub_df, model)
                    #import pdb;pdb.set_trace()

                    sub_df_q = sub_df[sub_df["question_class"].isin(question_name)]
                    result_mean:float= sub_df_q["is_correct"].mean()
    
                    bstp = bootstrap((sub_df_q["is_correct"],), np.mean, confidence_level=0.95,vectorized=True, batch=100, method='basic')
                    ci_at_95 = np.abs(bstp.confidence_interval.high -  result_mean)
              


                    result =  {"model":model, "dataset":dataset,"accuracy":f"{np.round(result_mean,round_to)} ({ci_at_95})"}
                    results.append(result)
                    dfs.append(sub_df)
            else:
                print(f"No results for {dataset}")

           
            #import pdb;pdb.set_trace()

    df_total = pd.concat(dfs, ignore_index=True)
    #df_total = df_total["is_correct"].mean().round(round_to)
    
    df_total_mean = df_total.groupby("model")["is_correct"].mean().round(round_to).reset_index()
    df_total_std  = df_total.groupby("model")["is_correct"].sem().round(round_to).reset_index()

    total_results = []
    for (_, row ),(_, row2 )in zip(df_total_mean.iterrows(),df_total_std.iterrows()):
        result =  {"model":row["model"], "dataset":"Total","accuracy":f"{np.round(row['is_correct']*multiplier,round_to)} ({np.round(row2['is_correct']*multiplier,round_to)})"}
        total_results.append(result)

    df_total = pd.DataFrame(total_results)
    df = pd.DataFrame(results)

    if tasks_metadata:
        for index, row in df.iterrows():
            dataset_name = row['dataset']
            metadata = tasks_metadata.get(dataset_name, {})
            for key, value in metadata.items():
                df.at[index, f'{key}_metadata'] = value

        pivot_df = df.pivot(index='task_name_metadata', columns='model', values=['accuracy'])

    else:
        pivot_df = df.pivot(index='dataset', columns='model', values=['accuracy'])

    pivot_df = pivot_df.T
    pivot_df.reset_index(inplace=True)
    
    return results,pivot_df



if __name__ == "__main__":


    round_to           = 2
    models:list[str]   = ["ALIGN","BLIP","OpenCLIP","BioMedCLIP","ConchCLIP","PLIP","QuiltCLIP","CogVLM","QwenVLM"]
    #models:list[str] = ["ALIGN","CLIP","BLIP","OpenCLIP","QuiltCLIP","OwlVIT2","PLIP","BioMedCLIP","ConchCLIP"]
    models:list[str] = ["ALIGN","BLIP","OpenCLIP","QuiltCLIP","PLIP","BioMedCLIP","ConchCLIP"]
    #models:list[str]   = ["ALIGN"]

    tasks_metadata = {
        "acevedo_et_al_2020":{"task_name":"White blood cell (BF)","synthetic":False,"num_classes": 8},
        "burgess_et_al_2024_contour":{"task_name":"Cell contour (S)","synthetic":True,"num_classes": 3},
        "burgess_et_al_2024_eccentricity":{"task_name":"Cell eccentricity (S)","synthetic":True,"num_classes": 3},
        "burgess_et_al_2024_texture":{"task_name":"Cell texture","synthetic (S)":True,"num_classes": 3},
        "empiar_sbfsem":{"task_name":"Organisms and structures in EM","synthetic":True,"num_classes": 5},
        "colocalization_benchmark":{"task_name":"Colocalization patterns","synthetic":True,"num_classes": 4},
        "eulenberg_et_al_2017_brightfield":{"task_name":"Cell cycle phase (BF)","synthetic":False,"num_classes":  7},
        "eulenberg_et_al_2017_darkfield":{"task_name":"Cell cycle phase (DF)","synthetic":False,"num_classes":  7},
        "eulenberg_et_al_2017_epifluorescence":{"task_name":"Cell cycle phase (EF)","synthetic":False,"num_classes":  5},
        "held_et_al_2010_galt":{"task_name":"Golgi morphology","synthetic":False,"num_classes":  8},
        "held_et_al_2010_h2b":{"task_name":"Cell cycle phase","synthetic":False,"num_classes":  9},
        "held_et_al_2010_mt":{"task_name":"Microtubule morphology","synthetic":False,"num_classes":  6},
        "hussain_et_al_2019":{"task_name":"Pre-cancerous and cervical cancer lesions","synthetic":False,"num_classes":  4},
        "icpr2020_pollen":{"task_name":"Pollen","synthetic":False,"num_classes":  4},
        "jung_et_al_2022":{"task_name":"White blood cellc (S)","synthetic":True,"num_classes": 5},
        "kather_et_al_2016":{"task_name":"colorectal cancer texture (a)","synthetic":False,"num_classes": 8},
        "kather_et_al_2018":{"task_name":"colorectal cancer texture (b)","synthetic":False,"num_classes": 8},
        "kather_et_al_2018_val7k":{"task_name":"colorectal cancer texture (c)","synthetic":False,"num_classes": 8},
        "nirschl_et_al_2018":{"task_name":"clinical chronic heart failure","synthetic":False,"num_classes": 2},
        "nirschl_unpub_fluorescence":{"task_name":"organisms and labeled structure","synthetic":False,"num_classes": 13},
        "tang_et_al_2019":{"task_name":"amyloid beta morphology patterns (a)","synthetic":False,"num_classes": 4},
        "wong_et_al_2022":{"task_name":"amyloid beta morphology patterns (b)","synthetic":False,"num_classes": 4},
        "wu_et_al_2023":{"task_name":"Mitochondrial morphology in (CryoET)","synthetic":False,"num_classes": 2},
        }

    datasets:list[str] = tasks_metadata.keys()

    extension:str = ".csv"
    filter_dict:dict[str,list[str]] = {"modality":["light microscopy"],"domain":["pathology"]}
    filter_dict:dict[str,list[str]] = None
    filenmae:str = "stats"
    
    if filter_dict:
        filenmae =construct_filter_name(filenmae,filter_dict) 
  
   
    question_columns = ["classification_0","classification_1"]
    question_column_str = "_".join(question_columns)
    output_path:str    = pathlib.Path("outputs","tables", filenmae + question_column_str)
    results,pivot_df = process_models_question_only(
        models=models,
        datasets=datasets,
        filter_dict = filter_dict,
        tasks_metadata =tasks_metadata,
        question_name=question_columns)
    save_table_to_latex_and_csv(pivot_df,output_path)

    
   

    import pdb; pdb.set_trace()



