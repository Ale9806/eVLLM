import pathlib
import ast
import pandas as pd
import os
import numpy as np
import json


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

CLIP_MODELS:list[str] = ["ALIGN","CLIP","BLIP","OpenCLIP","QuiltCLIP","OwlVIT2","PLIP","BioMedCLIP","ConchCLIP"]


def extract_top_k_element(cell_value,k=1):
    """
    Extracts the top k elements from a list-like string representation.

    Parameters:
    cell_value (str): A string representing a list of elements.
    k (int, optional): The number of top elements to extract. Defaults to 1.

    Returns:
    list: A list containing the top k elements from the input.

    Examples:
    >>> extract_top_k_element('[1, 2, 3, 4, 5]', 3)
    [1, 2, 3]
    """
    return eval(cell_value)[0:k]


def is_correct(correct_answer, top_k):
    """
    Checks if the correct answer is among the top k elements.

    Parameters:
    correct_answer: The correct answer to be checked.
    top_k: A list of top k elements.

    Returns:
    int: 1 if the correct answer is among the top k, otherwise 0.

    Examples:
    >>> is_correct(3, [1, 2, 3, 4, 5])
    1
    >>> is_correct(6, [1, 2, 3, 4, 5])
    0
    """
    return 1 if correct_answer in top_k else 0


def save_table_to_latex_and_csv(
    dataset_metadata_df: pd.DataFrame, 
    name: str
):
    """Save dataset metadata to CSV and LaTeX files.

    Parameters
    ----------
    dataset_metadata_df : pandas.DataFrame
        DataFrame containing dataset metadata.
    name : str, optional
        Name prefix for the output files (default is "images_per_dataset").

    Returns
    -------
    None
    """
    #import pdb;pdb.set_trace()
    # Create directory if it doesn't exist
    save_path = name.parents[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    field_csv_path = os.path.join(f"{name}.csv")
    field_latex_path = os.path.join(f"{name}.tex")

    # Save DataFrame to CSV and LaTeX files
    dataset_metadata_df.to_csv(field_csv_path, index=False)
    dataset_metadata_df.to_latex(field_latex_path, index=False)


def save_dict_to_json(
    data: dict, 
    name: str
):
    """Save dataset metadata to CSV and LaTeX files.

    Parameters
    ----------
    json : pandas.DataFrame
        DataFrame containing dataset metadata.
    name : str, optional
        Name prefix for the output files (default is "images_per_dataset").

    Returns
    -------
    None
    """
    #import pdb;pdb.set_trace()
    # Create directory if it doesn't exist

    with open(f"{name}.json", 'w') as f:
        json.dump(data, f)
  

  



def check_prediction(row):
    abcd           = ["A","B","C","D","E","F","G"]
    correct_idx    = row["correct_idx"]
    correct_letter = abcd[correct_idx]
    correct_answer = row['correct_answer']
    prediction     = row['prediction']
    #import pdb;pdb.set_trace()
    if correct_answer in prediction or correct_letter  in prediction[0:1]:
        return True
    else:
        return False


def filter_dataset(df:pd.DataFrame,filter_dict:dict[str,list]):
    append = False
    for column_name,filter_value in filter_dict.items():
        if column_name in df.columns:
            df = df.loc[df[column_name].isin(filter_value)]

    if len(df) > 0:
        append = True

    return df,append


def construct_filter_name(filenmae:str,filter_dict:dict[str,list[str]]) -> str:
    filename =filenmae
    for column_name,filter_values in filter_dict.items():
        filename += "_" + column_name
        for filter_value in filter_values:
            filename += "_" + str(filter_value)

    filename = filename.replace(" ","_")
    return filename


def replace_nan(df):
    df["model_answers"] = df["model_answers"].str.replace("nan","np.nan")
  

def get_results(df:pd.DataFrame, model:str) -> None:
    #try:
    if model not in CLIP_MODELS:
    #import pdb;pdb.set_trace()
        replace_nan(df)
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["text"], axis=1)
        df['is_correct'] = df.apply(check_prediction, axis=1)

    else: 
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["pred"][0] , axis=1)
        df["is_correct"] = 1*(df["correct_idx"] == df["prediction"])
    #except:
    #    import pdb;pdb.set_trace()



def get_confidence(row):
    probs = ast.literal_eval(row['model_answers'])['probs'][0]
    predicted_class = ast.literal_eval(row['model_answers'])['pred'][0]
    return probs[predicted_class]

