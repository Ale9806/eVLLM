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
from eval_utils import tasks_metadata

def process_models(
    models: list[str], 
    datasets: list[str], 
    round_to: int = 4, 
    extension: str = ".csv",
    filter_dict:dict[str,list]=None,
    to_percentage:bool=True) -> list[dict[str]]:
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

    if to_percentage:
        multiplier = 100
    else:
        multiplier = 1
    results = []
    save_results = {}
    datasets = list(datasets)
    datasets.remove("eulenberg_et_al_2017_darkfield")
    datasets.remove("eulenberg_et_al_2017_epifluorescence")

    for model in models:
        dfs = []
        print(f"Trying: {model}")
        
   
        for dataset in datasets:
           
                

            append = True
            #import pdb;pdb.set_trace()
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
                #words = ["empty", "debris", "background", "synthetic noise", "unif test slide"]
                #pattern = '|'.join(words)
                #filtered_df = sub_df[sub_df["correct_answer"].str.contains(pattern, case=False, na=False)]
                #drop = filtered_df["image_id"].unique()
                #sub_df = sub_df[~sub_df["image_id"].isin(drop)]
              
      
            
                ### Filter ###
                #if filter_dict:
                #    sub_df, append = filter_dataset(sub_df,filter_dict)
                    
                
                if append:
                    #print(f"added {dataset}")
                    dfs.append(sub_df)
            else:
                pass
                #print(f"No results for {dataset}")
        
        df = pd.concat(dfs, ignore_index=True)
        #import pdb;pdb.set_trace()
        get_results(df, model)
       # df["is_correct"] = df["is_correct"]*multiplier

      

        

        #result = df.groupby("question_class")["is_correct"].mean().to_dict()
        #result["dataset_total"] = df["is_correct"].mean()
        result = {}
        groups = {"domain":["domain","domain_1"],
                  "modality":["modality","modality_1"],
                  "stain":["stain","stain_1"],
                  "subdomain":["subdomain","subdomain_1"],
                  "submodality":["submodality","submodality_1"],
                  "total":["domain_1","modality_1","stain_1","subdomain_1","submodality_1","domain","modality","stain","subdomain","submodality"]
                     }       
        
        
        for key,group in groups.items():
            df_filtered, _ = filter_dataset(df,{"question_class":group})
            mean           =  df_filtered["is_correct"].mean()
            bstp = bootstrap((df_filtered["is_correct"],), np.mean, confidence_level=0.95, vectorized=True, batch=100, method='basic')
            ci_at_95            = np.abs(bstp.confidence_interval.high -   mean)
            save_results[model] = {group[0]:{"accuracy":mean,"se":bstp.standard_error},"ci95":ci_at_95 }
            result[key] = f"{np.round(mean,round_to)} ({ci_at_95})"

     
    
        result["model"] = model
        results.append(result)
   
    
    return results,save_results



if __name__ == "__main__":
    round_to           = 4
    models:list[str]   = ["GptApi","OpenCLIP_lion","OpenCLIP","QuiltCLIPRobust","PLIPRobust","QwenVLM","Random_model","CLIP","PaliGemma","CogVLM","ALIGN","BLIP","QuiltCLIP","PLIP","BioMedCLIP","ConchCLIP"]
    extension:str = ".csv"
    filter_dict:dict[str,list[str]] = None
    filenmae:str = "eval_without_eulenberg"
    if filter_dict:
        filenmae =construct_filter_name(filenmae,filter_dict) 


    ####### Instance Identification ##########
    #########################################
    datasets:list[str] = tasks_metadata.keys()
    output_path:str    = pathlib.Path("outputs","tables",filenmae )
    results,save_results = process_models(models=models, 
                               datasets=datasets,
                               filter_dict = filter_dict)

    print(f"Save results at:{output_path}")
    df_result = pd.DataFrame(results)
    model_column = df_result['model']
    df_result.drop(columns=['model'], inplace=True)
    df_result.insert(0, 'model', model_column)
    save_table_to_latex_and_csv(df_result,output_path)
    save_dict_to_json(save_results,output_path)


  

  
    import pdb; pdb.set_trace()



