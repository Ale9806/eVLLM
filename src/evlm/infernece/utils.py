import os
import pathlib
import pandas as pd

def unwrap_question_from_datapoint(data_point:dict[str,str]) -> dict[str,str]:
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
    #import pdb;pdb.set_trace()
    output_dir:pathlib.Path = output_name.parents[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    data  = pd.DataFrame(data)
    data.to_csv(str(output_name) +extension)
    print(f"Save output @ {output_name}")


    
