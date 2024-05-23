import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D

def save_plot(plt,output_name:str):
    # Save the plot as PDF and PNG with 300 DPI
    pdf_path = f"outputs/plots/{output_name}.pdf"
    png_path = f"outputs/plots/{output_name}.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()

    # Close the plot
    
    print(f"Plot saved as {pdf_path} and {png_path}")

def normalize_sizes(model_sizes):
    total_size = sum(model_sizes.values())
    normalized_sizes = {model:(size / total_size)*10000 for model, size in model_sizes.items()}
    return normalized_sizes


model_sizes = {
    "CLIP": 13000000000,
    "ALIGN": 1800000000,
    "CoCa": 4800000000,
    "OpenCLIP": 13000000000,
    "BLIP": 14000000,
    "PLIP": 208400,
    "QuiltNet": 1000000,
    "BioMedCLIP": 15000000,
    "CONCH": 1170000,
    "CogVLM": 1500000000,
    "QwenVLM": 1400000000,
    "Random": 1
}


normalized_sizes = normalize_sizes(model_sizes)
#import pdb;pdb.set_trace()
# Load the CSV into a DataFrame
data = {
    "model_name": ["ALIGN", "QuiltNet", "OpenCLIP", "BLIP", "PLIP", "BioMedCLIP", "CogVLM", "QwenVLM", "CoCa","CONCH", "Random"],
    "total_params": [172117841, 151277313, 151277313, 223744258, 151277313, 195902721, 17639685376, 9656935168, 8639685376,395232769, 0],
    "accuracy": [0.44, 0.35, 0.36, 0.24, 0.35, 0.49, 0.525, 0.555, 0.37, 0.19,0.19]
}
df = pd.DataFrame(data)


# Define colors for different 
colors = {
    # Group 1
    "CogVLM": "skyblue",
    "QwenVLM": "dodgerblue",

    # Group 2
    "ALIGN": "cornflowerblue",
    "BLIP": "navy",
    "CoCa": "deepskyblue",
    "OpenCLIP": "mediumturquoise",

    # Group 3
    "BioMedCLIP": "limegreen",

    # Group 4
    "PLIP": "violet",
    "QuiltNet": "indigo",
    "CONCH": "blueviolet",


    "Random": "dimgray"
}

markers = {
    # Group 1
    "CogVLM": "o",
    "QwenVLM": "o",

    # Group 2
    "ALIGN": "s",
    "BLIP": "s",
    "CoCa": "s",
    "OpenCLIP": "s",

    # Group 3
    "BioMedCLIP": "p",

    # Group 4
    "PLIP": "h",
    "QuiltNet": "h",
    "CONCH": "h",

    "Random": ">"
}


# Plotting
plt.figure(figsize=(10, 6))

# Baseline (Random)
plt.axhline(y=0.19, color=colors['Random'], linestyle='--', label='Random')

# Other models
for group in colors.keys():
    if group not in ['Random']:
        plt.scatter(df[df['model_name'] == group]['total_params'], df[df['model_name'] == group]['accuracy'],marker=markers[group], c=colors[group],s=normalized_sizes[group], label=group)


# Plot settings
plt.xscale('log')
plt.xlabel('Log Model Weights')
plt.ylabel('Accuracy')
plt.legend()
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def updatescatter(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([64])

def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(8)

plt.legend(bbox_to_anchor=(1, 0.5),loc='center left',handler_map={PathCollection : HandlerPathCollection(update_func=updatescatter),
                        plt.Line2D : HandlerLine2D(update_func = updateline)})

plt.tight_layout()
save_plot(plt,output_name="performance_vs_weight")