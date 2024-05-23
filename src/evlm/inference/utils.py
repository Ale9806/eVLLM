import os
import pathlib
import pandas as pd

def unwrap_question_from_datapoint(data_point:dict[str,str]) -> dict[str,str]:
    """
    Extracts questions from a data point.

    Parameters:
    data_point (dict[str, str]): The data point containing questions.

    Returns:
    dict[str, str]: Extracted questions.
    """
    questions = {}
    for question_type  in  ['microscopy_modality', 'microscopy_domain', 'classification']:
        if question_type  == 'microscopy_modality':
            for keys in ["modality","submodality"]:
                questions[keys] = data_point["questions"]['microscopy_modality'][keys]

        elif question_type  == 'microscopy_domain':
            for keys in ["domain","subdomain"]:
                questions[keys] = data_point["questions"]['microscopy_domain'][keys]
        else:
            questions[question_type] = data_point["questions"][question_type][question_type]
    return questions


def save_output(data,output_name:pathlib.Path,extension:str=".csv") -> None:
    """
    Saves data to a file.

    Parameters:
    data (list[dict[str, str]]): Data to be saved.
    output_name (pathlib.Path): Path to save the output file.
    extension (str): File extension. Default is ".csv".

    Returns:
    None
    """

    output_dir:pathlib.Path = output_name.parents[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    data  = pd.DataFrame(data)
    data.to_csv(str(output_name) +extension)
    print(f"Save output @ {output_name}")


def init_sub_results(data_point,add_synonyms:bool=False) -> dict[str,str]:
    """
    Initializes sub results.

    Parameters:
    data_point: Data point.
    add_synonyms (bool): Whether to include synonyms.

    Returns:
    dict[str, str]: Initialized sub results.
    """
    sub_results = {}
    meta_data:list[str] = ['microns_per_pixel',"domain","subdomain","modality","submodality","normal_or_abnormal"]

    if add_synonyms:
        meta_data = meta_data + ["label_description","label_synonyms"]
        
    for key in meta_data:
        sub_results[key] = data_point["custom_metadata"].get(key,"none")


    return sub_results


def add_alphabet(options:list[str],answer:str):
    """
    Adds alphabets to options.

    Parameters:
    options (list[str]): Options to add alphabets to.
    answer (str): Correct answer.

    Returns:
    tuple[list[str], str]: Alphabets added options and answer.
    """
    idx_to_option = {0:"A) ",1:"B) ",2:"C) ",3:"D) ",4:"E) ",5:"F) ",6:"G) ",7:"H) "}
    index = options.index(answer) 
    for i in range(0,len(options)):
        options[i] = idx_to_option[i] + options[i]

    answer =idx_to_option[index] + answer
    assert answer in options

    return options ,answer