import pathlib
import ast
import pandas as pd
import os

def extract_top_k_element(cell_value,k=1):
    return eval(cell_value)[0:k]

def is_correct(correct_answer, top_k):
    return 1 if correct_answer in top_k else 0

def save_table_to_latex_and_csv(
    dataset_metadata_df: pd.DataFrame, name: str
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

    
round_to           = 3
models:list[str]   = ["ALIGN","BLIP","OpenCLIP","BioMedCLIP","ConchCLIP","PLIP","QuiltCLIP","CogVLM","QwenVLM"]
datasets:list[str] =[
    'acevedo_et_al_2020', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', 'icpr2020_pollen',
    'nirschl_et_al_2018', 'jung_et_al_2022', 'wong_et_al_2022',
    'hussain_et_al_2019', 'colocalization_benchmark', 'kather_et_al_2016',
    'tang_et_al_2019', 'eulenberg_et_al_2017_brightfield',
    'burgess_et_al_2024_contour', 'nirschl_unpub_fluorescence',
    'burgess_et_al_2024_eccentricity', 'burgess_et_al_2024_texture',
    'held_et_al_2010']

output_path:str    = pathlib.Path("outputs","tables","eval" )
extension:str      = ".csv"
results  = []

for model in models:
    dfs = []
    for dataset in datasets: 
        data_path  = pathlib.Path("outputs",model,dataset + extension )
        if os.path.exists(data_path):
            sub_df        = pd.read_csv(data_path)
            dfs.append(sub_df)
        else:
            print(f"No results for {dataset}")


    df = pd.concat(dfs, ignore_index=True)
    import pdb;pdb.set_trace()

    if model in ["CogVLM","QwenVLM"]:
       #import pdb;pdb.set_trace()
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["text"].lower() , axis=1)
        df["correct_answer"] = df["correct_answer"].str.lower()
        df['is_correct'] = df.apply(check_prediction, axis=1)

    else: 
        df["prediction"] = df.apply(lambda row: eval(row["model_answers"])["pred"][0] , axis=1)
        df["is_correct"] = 1*(df["correct_idx"] == df["prediction"]) 

   
    result           = df.groupby("question_class")["is_correct"].mean().round(round_to).to_dict()
    result["dataset_total"]     = df["is_correct"].mean().round(round_to)
    result_std                  = df.groupby("question_class")["is_correct"].std().round(round_to).to_dict()
    result_std["dataset_total"] = df["is_correct"].std().round(round_to)

    for key ,value in result.items():
        result[key] = f"{result[key]} ({result_std[key]})"

    result["model"]         = model

  
    
    results.append(result)

df_result = pd.DataFrame(results)
model_column = df_result['model']
df_result.drop(columns=['model'], inplace=True)
df_result.insert(0, 'model', model_column)
save_table_to_latex_and_csv(df_result,output_path)

#import pdb; pdb.set_trace()x