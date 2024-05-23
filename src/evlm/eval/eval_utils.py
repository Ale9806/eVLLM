import pathlib
import ast
import pandas as pd
import os
import numpy as np

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


def check_prediction(row):
    correct_answer = row['correct_answer']
    prediction = row['prediction']
    
    if correct_answer in prediction or correct_answer[:3] in prediction:
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


def get_results(df:pd.DataFrame, model:str) -> None:
    if model not in ["ALIGN","QuiltCLIP","OwlVIT2","OpenCLIP","BLIP","PLIP","BioMedCLIP","ConchCLIP"]:
       #import pdb;pdb.set_trace()
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["text"].lower() , axis=1)
        df["correct_answer"] = df["correct_answer"].str.lower()
        df['is_correct'] = df.apply(check_prediction, axis=1)

    else: 
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["pred"][0] , axis=1)
        df["is_correct"] = 1*(df["correct_idx"] == df["prediction"]) 



def get_confidence(row):
    probs = ast.literal_eval(row['model_answers'])['probs'][0]
    predicted_class = ast.literal_eval(row['model_answers'])['pred'][0]
    return probs[predicted_class]

