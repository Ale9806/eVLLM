import pdb
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from scipy.stats import iqr

import sys
from pathlib import Path
module_path = str(Path(__file__).resolve().parent.parent) 
sys.path.append(module_path)

from eval.eval_utils import tasks_metadata,get_tasks_by_taxonomy,get_model_colors

#import pdb;pdb.set_trace()

def get_values_and_ci(df):
    # Separate values and errors
    for col in df.columns[1:]:
        df[[col + '_value', col + '_error']] = df[col].str.extract(r'([0-9.]+)\s*\((0\.[0-9]+)\)').astype(float)

    # Drop original columns with combined values and errors
    df.drop(columns=[col for col in df.columns if not col.endswith('_value') and not col.endswith('_error') and not col.endswith('model')], inplace=True)

    # Set index to 'model' for easier plotting
    df.set_index('model', inplace=True)

    # Transpose for plotting
    df_values = df[[col for col in df.columns if col.endswith('_value')]].transpose()
    df_errors = df[[col for col in df.columns if col.endswith('_error')]].transpose()
    df_values.index = df_values.index.str.replace('_value', '')
    df_errors.index = df_errors.index.str.replace('_error', '')
    return df_values,df_errors


def create_and_save_radar_plot(
    data_frame, 
    order=None, 
    filename='figures/radar_plot', 
    color_area=0.001,
    remove_tick_names:bool=False,
    add_legend:bool=False):
    # Create a figure and axis for the radar plot
    categories = data_frame.index.to_list()
    values     = data_frame.to_numpy()
    colors     = get_model_colors()
   
    if order == None:
        models = data_frame.columns
    else:
        models = order

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Number of categories
    num_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    angles = np.append(angles, angles[0])
    categories.append(categories[0])

    # Make the radar plot
    for model in models:
        values = data_frame[model].to_numpy()
        values = np.append(values, values[0])

        ax.plot(angles, values, label=model,color = colors[model])
        if color_area:
            #if model == "Random_model":
            #    ax.fill(angles, values, alpha=0.4)
            ax.fill(angles, values, alpha=color_area)  # Adjust alpha as needed for transparency

    # Set the labels for each category
    #import pdb;pdb.set_trace()
    better_lengends = {"domain":"Domain",
                       "subdomain":"Subdomain",
                       "modality":"Modality",
                       "submodality":"Submodality",
                       "stain":"Staining \ntechnique",
                      "cell texture and morphology profiling":'Cell texture \n&  morphology\nprofiling',
                       'Cell/organism/structure type identification':'Biological\nentity\nclassification',
                       'Cell cycle and stage identification':'Cell cycle \n& stage\n classification',
                       'Distinguish normal vs. abnormal': 'Normal \nvs.\n abnormal',
                       'Single molecule imaging':  'Single molecule\n imaging',
                       'Interpretation of neoplastic histopathology': 'Interpretation of\nneoplastic histopathology',
                       'Interpretation of non-neoplastic histopathology': 'Interpretation of\nnon-neoplastic histopathology', 

                       'Pap smear grading (BF)':'Pap smear\ngrading\n(PAP)',
                       'Colorectal tissue [a] (BF)':'Colorectal tissue\n[a] (H&E)',
                       'Colorectal tissue [b] (BF)':'Colorectal tissue\n[b] (H&E)',
                       'Colorectal tissue [c] (BF)':'Colorectal tissue\n[c] (H&E)',
                       'Clinical chronic heart failure (BF)': 'Clinical\nchronic\nheart failure \n(H&E)',
                       'Amyloid morphology [a] (BF)':  'Amyloid morphology\n [a] (IHC)',
                       'Amyloid morphology [b] (BF)':  'Amyloid morphology\n [b] (IHC)'
                       }
    for i in range(0,len(categories)):
        if categories[i] in better_lengends.keys():
            categories[i] = better_lengends[categories[i]]

    ax.set_thetagrids(angles * 180 / np.pi, labels=categories, rotation=45, fontsize=10)

    # Set the title and legend
    
    if add_legend:
        plt.legend(loc='center right')
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1.2, 0.4), fontsize=13)
        for line in legend.get_lines():
            line.set_linewidth(4.0)


    ax.tick_params(pad=26)
    ax.set_ylim([0,100])
    if remove_tick_names:
        filename += "remove_thicks"
        ax.set_xticklabels([])
    # Save the figure to the specified filename

    if add_legend:
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
        plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight', bbox_extra_artists=[legend], format='pdf')

    else:
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight',)
        plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight', format='pdf')

    # Show the plot (optional)
    print(f"Saved as {filename}")
    plt.show()

# Example usage:
# create_and_save_radar_plot(data_frame, order=None, filename='figures/radar_plot', color_area=0.3)


def radial_plot_from_df(
    df:str, 
    output_path:str, 
    axis_to_remove_list:list[str]=None, 
    order_axis_list:list[str]=None,
    model_order:list[str]=None,
    transpose:bool=False,
    groups:dict=None):
    
    df , _  =get_values_and_ci(df)
    if groups:
        df_means = {}
        for group,tasks in groups.items():
            matching_indices = df.index[df.index.isin(tasks)]
            mean = df.loc[matching_indices].mean(0)
            df_means[group] = mean


        df = pd.DataFrame(df_means).T
  
    #import pdb;pdb.set_trace()
    df = df * 100
    if transpose:
        df = df.transpose()

    if axis_to_remove_list:
        for axis_to_remove in axis_to_remove_list:
            try:
                df.drop(axis_to_remove, inplace=True)
            except:
                print(f"Column {axis_to_remove} not in dataframe")
        
    if order_axis_list and  groups == None:
        df = df.reindex(order_axis_list)
    
    create_and_save_radar_plot(df, order=model_order, filename=output_path)



if __name__ == "__main__":
 

    MODELS = ["GptApi","ALIGN","BioMedCLIP","ConchCLIP"],["GptApi","BioMedCLIP","ConchCLIP","QuiltCLIP","PLIP"]
    for model in MODELS:
        #import pdb;pdb.set_trace()
        name_ = "_".join(model)
        model_order                   = model
        axis_to_remove_list:list[str] = ["dataset_total","classification"]
        order_axis_list:list[str]     = ["submodality","modality","domain","subdomain","stain"]
        csv_data                      = "outputs/tables/eval.csv"
        output_path                   = f"outputs/tables/eval_{name_}"
        df_pi = pd.read_csv(csv_data)
        radial_plot_from_df(df_pi, output_path, axis_to_remove_list, order_axis_list,model_order=model_order)

        model_order                   = model
        axis_to_remove_list:list[str] = ["dataset_total","classification"]
        order_axis_list:list[str]     = ["submodality","modality","domain","subdomain","stain"]
        csv_data                      = "outputs/tables/eval_pathology.csv"
        output_path                   = f"outputs/tables/eval_path_{name_}"
        df_pi = pd.read_csv(csv_data)
        radial_plot_from_df(df_pi, output_path, axis_to_remove_list, order_axis_list,model_order=model_order)

        
    
        groups = get_tasks_by_taxonomy(tasks_metadata)
        #import pdb;pdb.set_trace()
        order = groups['cell texture and morphology profiling'] \
            + groups['Cell/organism/structure type identification'] \
            + groups['Cell cycle and stage identification'] \
            +  groups['Distinguish normal vs. abnormal'] \
            + groups['Single molecule imaging']  \
            + groups['Interpretation of neoplastic histopathology'] \
            +groups['Interpretation of non-neoplastic histopathology']


        model_order                   = model
        axis_to_remove_list:list[str] = ["dataset_total","classification"]
        order_axis_list:list[str]     = order
        csv_data                      = "outputs/tables/evalclassification_1.csv"
        output_path                   = f"outputs/tables/classification_{name_}"
        df_pc = pd.read_csv(csv_data)
        df_pc.drop(['level_0'], axis = 1, inplace = True) 
        #import pdb;pdb.set_trace()
        radial_plot_from_df(df_pc, output_path, axis_to_remove_list, order_axis_list,model_order=model_order,transpose=False,groups=groups)
        
        order = groups['Interpretation of neoplastic histopathology'] \
            +groups['Interpretation of non-neoplastic histopathology']
        #import pdb;pdb.set_trace()
        model_order                   = model
        axis_to_remove_list:list[str] = ["dataset_total","classification"]
        order_axis_list:list[str]     = order
        csv_data                      = "outputs/tables/evalclassification_1.csv"
        output_path                   = f"outputs/tables/classification_path_{name_}"
        df_pc = pd.read_csv(csv_data)
        df_pc.drop(['level_0'], axis = 1, inplace = True) 
        #import pdb;pdb.set_trace()
        radial_plot_from_df(df_pc, output_path, axis_to_remove_list, order_axis_list,model_order=model_order,transpose=False)
        
        
        print("Done")
        
        
