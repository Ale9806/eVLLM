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

    for model in models:
        dfs = []
        #print(f"Trying: {model}")
        for dataset in datasets:
            append = True
            #import pdb;pdb.set_trace()
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
            
                ### Filter ###
                if filter_dict:
                    sub_df, append = filter_dataset(sub_df,filter_dict)
                    
                
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


def process_models_question_only(
    models: list[str], 
    datasets: list[str], 
    round_to: int = 4, 
    extension: str = ".csv",
    filter_dict:dict[str,list]=None,
    tasks_metadata:dict[str,dict]=None,
    to_percentage:bool=True,
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
    result_totals = []
    totals = []
    dfs = []
    for model in models:
        print(f'Model:{model}')
        result = {}
        dataset_total_ = []
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
                    if model not in CLIP_MODELS:
                        sub_df,append = filter_dataset(sub_df,{"question_class":["classification"]})
                    else:
                        sub_df,append = filter_dataset(sub_df,{"question_class":question_name})
                    
                    if append:
                        sub_df["dataset"] = dataset
                        sub_df["model"]   = model
                        #import pdb; pdb.set_trace()
                        try:
                            get_results(sub_df, model)
                            #sub_df["is_correct"] = sub_df["is_correct"]*multiplier
                        except:
                            import pdb; pdb.set_trace()

                        result_mean:float= sub_df["is_correct"].mean()
                    
                        bstp = bootstrap((sub_df["is_correct"],), np.mean, confidence_level=0.95,vectorized=True, batch=100, method='basic')
                        ci_at_95 = np.abs(bstp.confidence_interval.high -  result_mean)
                


                        result =  {"model":model, "dataset":dataset,"accuracy":f"{np.round(result_mean,round_to)} ({ci_at_95})"}
                        results.append(result)
                        dfs.append(sub_df)
                        dataset_total_.append(sub_df)
                    else:
                        print(f"Could not run:{model}:{dataset}")
            else:
                print(f"No results for {dataset}")
                
        df_total_dataset = pd.concat(dataset_total_, ignore_index=True)

        result_mean:float= df_total_dataset["is_correct"].mean()
        bstp = bootstrap((df_total_dataset["is_correct"],), np.mean, confidence_level=0.95,vectorized=True, batch=100, method='basic')
        ci_at_95 = np.abs(bstp.confidence_interval.high -  result_mean)
        result_t =  {"model":model, "dataset":"total","accuracy":f"{np.round(result_mean,round_to)} ({ci_at_95})"}
        result_totals.append(result_t)
      

           
            #import pdb;pdb.set_trace()

    df_total = pd.concat(dfs, ignore_index=True)
    #df_total = df_total["is_correct"].mean().round(round_to)
    
    #df_total_mean = df_total.groupby("model")["is_correct"].mean().round(round_to).reset_index()
    #df_total_std  = df_total.groupby("model")["is_correct"].sem().round(round_to).reset_index()

   # total_results = []
    #for (_, row ),(_, row2 )in zip(df_total_mean.iterrows(),df_total_std.iterrows()):
    #    result =  {"model":row["model"], "dataset":"Total","accuracy":f"{np.round(row['is_correct']*multiplier,round_to)} ({np.round(row2['is_correct']*multiplier,round_to)})"}
    #    total_results.append(result)

    #df_total = pd.DataFrame(total_results)
    df = pd.DataFrame(results)

    if tasks_metadata:
        for index, row in df.iterrows():
            dataset_name = row['dataset']
            metadata = tasks_metadata.get(dataset_name, {})
            for key, value in metadata.items():
                if key == "task_name":
                    value += " (" +  metadata["submodality"] + ")"
                df.at[index, f'{key}_metadata'] = value

        #import pdb;pdb.set_trace()
        pivot_df = df.pivot(index='task_name_metadata', columns='model', values=['accuracy'])

    else:
        pivot_df = df.pivot(index='dataset', columns='model', values=['accuracy'])

    pivot_df = pivot_df.T
    pivot_df.reset_index(inplace=True)
    
 
    return results,pivot_df,result_totals



def process_models_cognition_only(
    models: list[str], 
    datasets: list[str], 
    round_to: int = 4, 
    extension: str = ".csv",
    filter_dict:dict[str,list]=None,
    tasks_metadata:dict[str,dict]=None,
    to_percentage:bool=True,
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
        print(f'Model:{model}')
        result = {}
        for dataset in datasets:
            append = True
            data_path = pathlib.Path("outputs", model, dataset + extension)
            if os.path.exists(data_path):
                sub_df = pd.read_csv(data_path)
            
                ### Filter ###
                if filter_dict:
                    sub_df, append = filter_dataset(sub_df,filter_dict)
                    
                #import pdb;pdb.set_trace()
                if append:
                    #print(f"added {dataset}")
                    sub_df,append  = filter_dataset(sub_df,{"question_class":question_name})
                    sub_df["dataset"] = dataset
                    sub_df["model"]   = model
                    get_results(sub_df, model,prompt_type="question")
                    #sub_df["is_correct"] = sub_df["is_correct"]*multiplier
                    #import pdb;pdb.set_trace()

                    groups = ['cell biology', 'cell and molecular biology', 'botany', 'virology',
                            'neurobiology', 'developmental biology', 'microbiology',
                            'neuropathology', 'genetics', 'gastrointestinal pathology',
                        'immunology']

                    print(model)
                    for group in groups:
                       #import pdb;pdb.set_trace()
                        df_filtered, _ = filter_dataset(sub_df,{"subdomain":[group]})
                        mean           =  df_filtered["is_correct"].mean()
                        #import pdb;pdb.set_trace()
                        try:
                            bstp = bootstrap((df_filtered["is_correct"],), np.mean, confidence_level=0.95, vectorized=False, method='basic')
                        except:
                            pass

                        ci_at_95            = np.abs(bstp.confidence_interval.high -   mean)
                        result[group] = f"{np.round(mean,round_to)} ({ci_at_95})"
                        
                    bstp     = bootstrap((sub_df["is_correct"],), np.mean, confidence_level=0.95, vectorized=False, method='basic')
                    mean     =  sub_df["is_correct"].mean()
                    ci_at_95 = np.abs(bstp.confidence_interval.high -   mean)
                    result["total"] = f"{np.round(mean,round_to)} ({ci_at_95})"
                    #import pdb;pdb.set_trace()

                    
                    
                    result["model"] = model
                    results.append(result)
                    #import pdb; pdb.set_trace()
            
                  
                  
    
 
    return results



if __name__ == "__main__":

    #"GptApi","
    round_to           = 4
    models:list[str]  = ["GptApi","OpenCLIP_lion","OpenCLIP","QuiltCLIPRobust","PLIPRobust","QwenVLM","Random_model","CLIP","PaliGemma","CogVLM","ALIGN","BLIP","QuiltCLIP","PLIP","BioMedCLIP","ConchCLIP"]
    #models:list[str] = ["Random_model"]
    extension:str = ".csv"
    #filter_dict:dict[str,list[str]] = {"modality":["light microscopy"],"domain":["pathology"]}
    filter_dict:dict[str,list[str]] = None
    filenmae:str = "eval"
    if filter_dict:
        filenmae =construct_filter_name(filenmae,filter_dict) 

    #########################################
    ####### ANALYZE Cogntion ##########
    #########################################
    datasets = ["cognition"]
    print("Running classifcation Eval")
    question_columns = [f"cognition_{i}" for i in range(0,100)]
    #question_columns = ["cognition"]
    question_column_str = "cognition"
    output_path:str    = pathlib.Path("outputs","tables", filenmae + question_column_str)
   
    results= process_models_cognition_only(
        models=models,
        datasets=datasets,
        filter_dict = filter_dict,
        tasks_metadata =tasks_metadata,
        question_name=question_columns)
    df_result = pd.DataFrame(results)
    model_column = df_result['model']
    df_result.drop(columns=['model'], inplace=True)
    df_result.insert(0, 'model', model_column)
    save_table_to_latex_and_csv(df_result,output_path)

    #########################################
    ####### Instance Identification (H&E) ##########
    #########################################
    #datasets:list[str] = tasks_metadata.keys()

    #import pdb;pdb.set_trace()
    output_path:str    = pathlib.Path("outputs","tables",filenmae + "HandE")
    results,save_results = process_models(models=models, 
                                           datasets=['kather_et_al_2016', 'kather_et_al_2018', 'kather_et_al_2018_val7k', 'nirschl_et_al_2018'],
                                           filter_dict = filter_dict)

    print(f"Save results at:{output_path}")
    df_result = pd.DataFrame(results)
    model_column = df_result['model']
    df_result.drop(columns=['model'], inplace=True)
    df_result.insert(0, 'model', model_column)
    save_table_to_latex_and_csv(df_result,output_path)
    save_dict_to_json(save_results,output_path)


    output_path:str    = pathlib.Path("outputs","tables",filenmae +"_pathology" )
    results,save_results = process_models(models=models, 
                                            datasets=["hussain_et_al_2019","acevedo_et_al_2020","jung_et_al_2022",'kather_et_al_2016', 'kather_et_al_2018', 'kather_et_al_2018_val7k', 'nirschl_et_al_2018'],
                                           filter_dict = filter_dict)

    print(f"Save results at:{output_path}")
    df_result = pd.DataFrame(results)
    model_column = df_result['model']
    df_result.drop(columns=['model'], inplace=True)
    df_result.insert(0, 'model', model_column)
    save_table_to_latex_and_csv(df_result,output_path)
    save_dict_to_json(save_results,output_path)



   

    #########################################
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


    #########################################
    ####### ANALYZE CLASSIFICATION ##########
    #########################################


    print("Running classifcation Eval path:")
    question_columns = ["classification_1"]
    question_column_str = "_".join(question_columns) + "HandE"
    output_path:str    = pathlib.Path("outputs","tables", filenmae + question_column_str)
    results,pivot_df,result_totals= process_models_question_only(
        models=models,
        datasets=['kather_et_al_2016', 'kather_et_al_2018', 'kather_et_al_2018_val7k', 'nirschl_et_al_2018'],
        filter_dict = filter_dict,
        tasks_metadata =tasks_metadata,
        question_name=question_columns)
    save_table_to_latex_and_csv(pivot_df,output_path)


    df_result = pd.DataFrame(result_totals)
    model_column = df_result['model']
    df_result.drop(columns=['model'], inplace=True)
    df_result.insert(0, 'model', model_column)
    save_table_to_latex_and_csv(df_result,Path(str(output_path) + "total"))
    


    print("Running classifcation Eval path:")
    question_columns = ["classification_1"]
    question_column_str = "_".join(question_columns) + "_pathology"
    output_path:str    = pathlib.Path("outputs","tables", filenmae + question_column_str)
    results,pivot_df,result_totals= process_models_question_only(
        models=models,
        datasets=["hussain_et_al_2019","acevedo_et_al_2020","jung_et_al_2022",'kather_et_al_2016', 'kather_et_al_2018', 'kather_et_al_2018_val7k', 'nirschl_et_al_2018'],
        filter_dict = filter_dict,
        tasks_metadata =tasks_metadata,
        question_name=question_columns)
    save_table_to_latex_and_csv(pivot_df,output_path)

    df_result = pd.DataFrame(result_totals)
    model_column = df_result['model']
    df_result.drop(columns=['model'], inplace=True)
    df_result.insert(0, 'model', model_column)
    save_table_to_latex_and_csv(df_result,Path(str(output_path) + "total"))
    


    print("Running classifcation Eval")
    question_columns = ["classification_0","classification_1"]
    question_column_str = "_".join(question_columns)
    output_path:str    = pathlib.Path("outputs","tables", filenmae + question_column_str)
    results,pivot_df,result_totals= process_models_question_only(
        models=models,
        datasets=datasets,
        filter_dict = filter_dict,
        tasks_metadata =tasks_metadata,
        question_name=question_columns)
    save_table_to_latex_and_csv(pivot_df,output_path)

    for question_column in question_columns:
        question_column_str = question_column
        output_path:str     = pathlib.Path("outputs","tables", filenmae + question_column_str)
        results,pivot_df,result_totals = process_models_question_only(
            models=models,
            datasets=datasets,
            filter_dict = filter_dict,
            tasks_metadata =tasks_metadata,
            question_name=[question_column])
        save_table_to_latex_and_csv(pivot_df,output_path)

        df_result = pd.DataFrame(result_totals)
        model_column = df_result['model']
        df_result.drop(columns=['model'], inplace=True)
        df_result.insert(0, 'model', model_column)
        save_table_to_latex_and_csv(df_result,Path(str(output_path) + "total"))
        


    print("DONE :)")

    import pdb; pdb.set_trace()



