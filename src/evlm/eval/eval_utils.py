import pathlib
import ast
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import string
tasks_metadata = {
    'acevedo_et_al_2020': {
        'task_name': 'White blood cell',
        'synthetic': False,
        'num_classes': 8,
        'submodality': 'BF',
        'taxonomy': 'Cell/organism/structure type identification'
    },
    'burgess_et_al_2024_contour': {
        'task_name': 'Cell contour',
        'synthetic': True,
        'num_classes': 3,
        'submodality': 'S',
        'taxonomy': 'cell texture and morphology profiling'
    },
    'burgess_et_al_2024_eccentricity': {
        'task_name': 'Cell eccentricity',
        'synthetic': True,
        'num_classes': 3,
        'submodality': 'S',
        'taxonomy': 'cell texture and morphology profiling'
    },
    'burgess_et_al_2024_texture': {
        'task_name': 'Cell texture',
        'synthetic': True,
        'num_classes': 3,
        'submodality': 'S',
        'taxonomy': 'cell texture and morphology profiling'
    },
    'empiar_sbfsem': {
        'task_name': 'Organisms and structures in EM',
        'synthetic': False,
        'num_classes': 5,
        'submodality': 'SBSEM',
        'taxonomy': 'Distinguish normal vs. abnormal'
    },
    'colocalization_benchmark': {
        'task_name': 'Molecule colocalization',
        'synthetic': True,
        'num_classes': 4,
        'submodality': 'S',
        'taxonomy': 'Single molecule imaging'
    },
    'eulenberg_et_al_2017_brightfield': {
        'task_name': 'Cell cycle phase',
        'synthetic': False,
        'num_classes': 7,
        'submodality': 'BF',
        'taxonomy': 'Cell cycle and stage identification'
    },
    'eulenberg_et_al_2017_darkfield': {
        'task_name': 'Cell cycle phase',
        'synthetic': False,
        'num_classes': 7,
        'submodality': 'DF',
        'taxonomy': 'Cell cycle and stage identification'
    },
    'eulenberg_et_al_2017_epifluorescence': {
        'task_name': 'Cell cycle phase',
        'synthetic': False,
        'num_classes': 5,
        'submodality': 'EF',
        'taxonomy': 'Cell cycle and stage identification'
    },
    'held_et_al_2010_galt': {
        'task_name': 'Golgi morphology',
        'synthetic': False,
        'num_classes': 8,
        'submodality': 'EF',
        'taxonomy': 'Cell cycle and stage identification'
    },
    'held_et_al_2010_h2b': {
        'task_name': 'Cell cycle (Chromatin)',
        'synthetic': False,
        'num_classes': 9,
        'submodality': 'EF',
        'taxonomy': 'Cell cycle and stage identification'
    },
    'held_et_al_2010_mt': {
        'task_name': 'Microtubule morphology',
        'synthetic': False,
        'num_classes': 6,
        'submodality': 'EF',
        'taxonomy': 'Cell cycle and stage identification'
    },
    'hussain_et_al_2019': {
        'task_name': 'Pap smear grading',
        'synthetic': False,
        'num_classes': 4,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of neoplastic histopathology'
    },
    'icpr2020_pollen': {
        'task_name': 'Pollen structures',
        'synthetic': False,
        'num_classes': 4,
        'submodality': 'BF',
        'taxonomy': 'Distinguish normal vs. abnormal'
    },
    'jung_et_al_2022': {
        'task_name': 'White blood cell',
        'synthetic': True,
        'num_classes': 5,
        'submodality': 'S',
        'taxonomy': 'Cell/organism/structure type identification'
    },
    'kather_et_al_2016': {
        'task_name': 'Colorectal tissue [a]',
        'synthetic': False,
        'num_classes': 8,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of neoplastic histopathology'
    },
    'kather_et_al_2018': {
        'task_name': 'Colorectal tissue [b]',
        'synthetic': False,
        'num_classes': 8,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of neoplastic histopathology'
    },
    'kather_et_al_2018_val7k': {
        'task_name': 'Colorectal tissue [c]',
        'synthetic': False,
        'num_classes': 8,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of neoplastic histopathology'
    },
    'nirschl_et_al_2018': {
        'task_name': 'Clinical chronic heart failure',
        'synthetic': False,
        'num_classes': 2,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of non-neoplastic histopathology'
    },
    'nirschl_unpub_fluorescence': {
        'task_name': 'Fluorescent Organisms/Structurese',
        'synthetic': False,
        'num_classes': 13,
        'submodality': 'TIRF',
        'taxonomy': 'Cell/organism/structure type identification'
    },
    'tang_et_al_2019': {
        'task_name': 'Amyloid morphology [a]',
        'synthetic': False,
        'num_classes': 4,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of non-neoplastic histopathology'
    },
    'wong_et_al_2022': {
        'task_name': 'Amyloid morphology [b]',
        'synthetic': False,
        'num_classes': 4,
        'submodality': 'BF',
        'taxonomy': 'Interpretation of non-neoplastic histopathology'
    },
    'wu_et_al_2023': {
        'task_name': 'Mitochondrial morphology',
        'synthetic': False,
        'num_classes': 2,
        'submodality': 'CET',
        'taxonomy': 'Distinguish normal vs. abnormal'
    }
}
CLIP_MODELS:list[str] = ["OpenCLIP_lion","QuiltCLIPRobust","PLIPRobust","ALIGN","CLIP","BLIP","OpenCLIP","QuiltCLIP","OwlVIT2","PLIP","BioMedCLIP","ConchCLIP","Random_model"]


def get_model_colors(verion=0):
    if verion==0:
        tab20_colors = plt.get_cmap('tab20').colors
        model_colors = {
            ## autoregressive generalist
            'CogVLM':"green",
            'QwenVLM':"red",
            'PaliGemma':"orange",
            "GptApi":"blue",
            "GPT-4o":"blue",
            
            ## contrastive generalist
            'ALIGN': tab20_colors[6],
            'BLIP': tab20_colors[4],
            'OpenCLIP': tab20_colors[8],
            "CLIP":tab20_colors[0],
        
            ## specialist 
            'BioMedCLIP': 'mediumpurple',
            'BiomedCLIP': 'mediumpurple',
            'QuiltCLIP': '#FFB6C1',
            'PLIP': '#FF69B4',
            'ConchCLIP': '#DDA0DD',
            'Conch': '#DDA0DD',
            
            ## random
            "Random":"gray",
            "Random_model":"gray",
        }
    else:
        raise ValueError()
        
    return model_colors 


def get_tasks_by_taxonomy(tasks_metadata,add_submodality = True):
    taxonomy_dict = {}
    
    for task in tasks_metadata.values():
        taxonomy = task['taxonomy']
        task_name = task['task_name']
        if add_submodality:
             task_name += " (" +  task["submodality"] + ")"


        #import pdb;pdb.set_trace()
        
        if taxonomy not in taxonomy_dict:
            taxonomy_dict[taxonomy] = []
        
        taxonomy_dict[taxonomy].append(task_name)
    
    return taxonomy_dict

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


def replace_nan(df,with_zero=False):
    if with_zero:
        df["model_answers"] = df["model_answers"].str.replace("nan","0")
    else:
        df["model_answers"] = df["model_answers"].str.replace("nan","np.nan")
  

def get_results(df:pd.DataFrame, model:str,prompt_type="caption") -> None:
    if model not in CLIP_MODELS:
        replace_nan(df)
        #try:
       # import pdb;pdb.set_trace()
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["text"], axis=1)
        df['is_correct'] = df.apply(check_prediction, axis=1)

    else: 
        #import pdb;pdb.set_trace()
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["pred"][0] , axis=1)
        if model == "Random_model":
            if prompt_type == "question":
                df['is_correct'] = 1*df.apply(lambda row: row.correct_answer in row.prediction, axis=1)
            else:
                df["is_correct"] = 1*(df["correct_answer"] == df["prediction"])
            
        else:
            df["is_correct"] = 1*(df["correct_idx"] == df["prediction"])
            
    


def get_confidence(row):

    #import pdb;pdb.set_trace()
    
    # Embedding models API
    if row["model"] in CLIP_MODELS or row["model"] in ["OpenCLIP_lion"]:
        #import pdb;pdb.set_trace()
        probs                 = ast.literal_eval(row['model_answers'])['probs'][0]
        pred                  = ast.literal_eval(row['model_answers'])['pred'][0]
        correct_idx           = row["correct_idx"]
        prob_at_correct_class = probs[correct_idx]


    # Auto Regressive models API
    else:
        correct_idx     = row["correct_idx"]
        correct_letter  = string.ascii_uppercase[correct_idx]
        prob_at_correct_class = ast.literal_eval(row["model_answers"])['probs_choices'][correct_letter]
        import pdb;pdb.set_trace()
        df["is_correct"] 
        

    return prob_at_correct_class



if __name__ == "__main__":
    groups =get_tasks_by_taxonomy(tasks_metadata)
    import pdb;pdb.set_trace()