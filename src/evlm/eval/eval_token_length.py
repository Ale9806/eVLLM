from pathlib import Path
import sys
import ast
import pandas as pd
import os
import numpy as np

module_path = str(Path(__file__).resolve().parent) 
sys.path.append(module_path)

from eval_utils import *
from plot_utils import *

def process_models(
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
        for dataset in datasets:
            append = True
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
            
                ### Filter ###
                if filter_dict:
                    sub_df, append = filter_dataset(sub_df,filter_dict)
                
                if append:
                    print(f"added {dataset}")
                    sub_df["model"] = model
                    dfs.append(sub_df)

            else:
                print(f"No results for {dataset}")

        df = pd.concat(dfs, ignore_index=True)
        df["char_length"] = df["correct_answer"].apply(len)
        df['confidence']  = df.apply(get_confidence, axis=1)
        get_results(df, model)

        #df  = df.groupby('question_class')["is_correct"].mean().round(round_to).reset_index()
        #accuracy_df = df.groupby('question_class')[["is_correct",'confidence']].mean().round(round_to).reset_index()
        for model in df["model"].unique():
            specific_model = df[df["model"]==model]
            plot_length_confidence(specific_model,f"{model}_length_confidence",model=model)
            plot_histogram(specific_model,f"{model}_confidence_histogram",model=model)
            plot_calibration(df,f"{model}_confidence_correctnes_correlation")

        scenarios = [["classification_0"], ["classification_1"], ['modality', 'submodality', 'domain', 'subdomain', 'stain']]

        for scenario in scenarios:
            temp_df = df[df["question_class"].isin(scenario)]
            temp_df['char_length_bin'] = pd.qcut(temp_df['char_length'], q=4)
            mean_is_correct_per_model_bin = temp_df.groupby(['model', 'char_length_bin'])['is_correct'].mean().reset_index()
            std_is_correct_per_model_bin = temp_df.groupby(['model', 'char_length_bin'])['is_correct'].std().reset_index()

            std_is_correct_per_model_bin  = std_is_correct_per_model_bin.round(2)
            mean_is_correct_per_model_bin = mean_is_correct_per_model_bin.round(2)
            mean_std_is_correct_per_model_bin = mean_is_correct_per_model_bin.merge(std_is_correct_per_model_bin,on=['model', 'char_length_bin'], suffixes=('_mean', '_std'))
            mean_std_is_correct_per_model_bin['is_correct_ms'] = mean_std_is_correct_per_model_bin.apply(lambda row: f"{row['is_correct_mean']} ({row['is_correct_std']})",axis=1)
            pivot_df = mean_std_is_correct_per_model_bin.pivot_table(index='char_length_bin', columns='model', values='is_correct_ms', aggfunc=','.join).reset_index()
            save_table_to_latex_and_csv(pivot_df,Path(f"outputs/tables/{ '_'.join(scenario)}"))

    import pdb;pdb.set_trace()

    return results





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
        for dataset in datasets:
            append = True
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
            
                ### Filter ###
                if filter_dict:
                    sub_df, append = filter_dataset(sub_df,filter_dict)
                
                if append:
                    print(f"added {dataset}")
                    sub_df["model"]    = model
                    sub_df["dataset"] = dataset
                    dfs.append(sub_df)

            else:
                print(f"No results for {dataset}")

            

        df = pd.concat(dfs, ignore_index=True)
        df["char_length"] = df["correct_answer"].apply(len)
        df['confidence']  = df.apply(get_confidence, axis=1)
        get_results(df, model)


        #df["question_class"]  == classification_0
        #df["dataset"]  =  df["question_class"] 
        df.loc[df["question_class"] == "classification_0", "question_class"] = df.loc[df["question_class"] == "classification_0", "dataset"]
        plot_calibration(df,f"{model}_confidence_correctnes_correlation")
       
        #accuracy_df = df.groupby('question_class')[["is_correct",'confidence']].mean().round(round_to).reset_index()
       
    import pdb;pdb.set_trace()

    return results



if __name__ == "__main__":


    round_to           = 2
    models:list[str]   = ["ALIGN","BLIP","OpenCLIP","BioMedCLIP","ConchCLIP","PLIP","QuiltCLIP","CogVLM","QwenVLM"]
    models:list[str] = ["ALIGN","BLIP","OpenCLIP","QuiltCLIP","PLIP","BioMedCLIP","ConchCLIP"]
    #models:list[str]   = ["ALIGN","BLIP","ConchCLIP"]
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
    filenmae:str = "eval"
    if filter_dict:
        filenmae =construct_filter_name(filenmae,filter_dict) 
  
    output_path:str    = pathlib.Path("outputs","tables",filenmae )

    results = accuracy_confidence_regression(models=models, 
                             datasets=datasets,
                             filter_dict = filter_dict)

   
    import pdb; pdb.set_trace()



