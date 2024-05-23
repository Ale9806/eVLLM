import pdb
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from scipy.stats import iqr



def create_and_save_radar_plot(data_frame, order=None, filename='figures/radar_plot', color_area=0.1):
    # Create a figure and axis for the radar plot
    categories = data_frame.index.to_list()
    values = data_frame.to_numpy()
   
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

        ax.plot(angles, values, label=model)
        if color_area:
            ax.fill(angles, values, alpha=color_area)  # Adjust alpha as needed for transparency

    # Set the labels for each category
    ax.set_thetagrids(angles * 180 / np.pi, labels=categories, rotation=45, fontsize=10)

    # Set the title and legend
    plt.legend(loc='center right')
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1.2, 0.4), fontsize=13)
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    ax.tick_params(pad=26)
    ax.set_ylim([0,100])
    # Save the figure to the specified filename
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight', bbox_extra_artists=[legend], format='pdf')

    # Show the plot (optional)
    plt.show()

# Example usage:
# create_and_save_radar_plot(data_frame, order=None, filename='figures/radar_plot', color_area=0.3)


def process_csv(
    csv_data:str, 
    output_path:str, 
    axis_to_remove_list:list[str]=None, 
    order_axis_list:list[str]=None,
    model_order:list[str]=None,
    transpose:bool=True):

    df = pd.read_csv(csv_data)
    df.iloc[:, 1:] = df.iloc[:, 1:].replace(r'\s*\([^)]*\)', '', regex=True) # Remove numbers in parentheses from all columns except the first one (model)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df.set_index('model', inplace=True)
    df = df * 100
    if transpose:
        df = df.transpose()

    if axis_to_remove_list:
        for axis_to_remove in axis_to_remove_list:
            try:
                df.drop(axis_to_remove, inplace=True)
            except:
                print(f"Column {axis_to_remove} not in dataframe")
        
    if order_axis_list:
        df = df.reindex(order_axis_list)

    create_and_save_radar_plot(df, order=model_order, filename=output_path)



model_order = ["ALIGN","BLIP","OpenCLIP","BioMedCLIP","ConchCLIP","QuiltCLIP","PLIP"]
axis_to_remove_list:list[str] = ["dataset_total","classification"]
order_axis_list:list[str]     = ["submodality","modality","domain","subdomain","stain"]
csv_data    = "outputs/tables/eval.csv"
output_path = "outputs/tables/eval"
process_csv(csv_data, output_path, axis_to_remove_list, order_axis_list,model_order=model_order)


axis_to_remove_list:list[str] = None
order_axis_list:list[str]     = None
csv_data    = "outputs/tables/evalclassification.csv"
output_path = "outputs/tables/evalclassification"
process_csv(csv_data, output_path, axis_to_remove_list, order_axis_list,model_order=model_order,transpose=False)