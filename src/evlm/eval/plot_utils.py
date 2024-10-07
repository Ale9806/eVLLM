import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import numpy as np


def save_plot(plt,output_name:str):
    # Save the plot as PDF and PNG with 300 DPI
    pdf_path = f"outputs/plots/{output_name}.pdf"
    png_path = f"outputs/plots/{output_name}.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()

    # Close the plot
     

    print(f"Plot saved as {pdf_path} and {png_path}")


def plot_calibration(df,output_name,bins:int=100,model:str=None):
    if model:
        plt.title(model)
   
    prob_true, prob_pred = calibration_curve(df["is_correct"], df["confidence"], n_bins=bins)
    b_score = brier_score_loss(df["is_correct"], df["confidence"])
    plt.title(f"Brier Score:{b_score}")
    plt.plot(prob_true, prob_pred,"bo")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    save_plot(plt,output_name)


def plot_histogram(df,output_name,bins:int=10,model:str=None):
    correct   = df[df["is_correct"] == 1]
    incorrect = df[df["is_correct"] == 0]
    
    if model:
        plt.title(model)
    plt.hist(correct["confidence"], bins=bins, color='blue', edgecolor='black',density=True)
    plt.axvline(x = correct["confidence"].mean() , color = 'blue',linestyle="--",label = 'mean')
    plt.hist(incorrect["confidence"], bins=bins, color='red', edgecolor='black',alpha=0.5,density=True)
    plt.axvline(x = incorrect["confidence"].mean() , color = 'red',linestyle="--",label = 'mean')
    
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.xlim([0,1])

    save_plot(plt,output_name)
    

def plot_length_confidence(df,output_name,model:str=None):
    correct   = df[df["is_correct"] == 1]
    incorrect = df[df["is_correct"] == 0]
    
    #fig, ax   = plt.subplots(figsize=(8, 6))
    #ax.scatter(correct["char_length"], correct["confidence"], color='blue', alpha=0.01,s=4)
    #ax.set_xlabel("Question length")
    #ax.set_ylabel("Confidence")
    #ax.set_ylim([0,1])
    #save_plot(plt,output_name + "correct")

    #fig, ax   = plt.subplots(figsize=(8, 6))
    #ax.scatter(incorrect["char_length"], incorrect["confidence"], color='red', alpha=0.01,s=4)
    #ax.set_xlabel("Question length")
    #ax.set_ylabel("Confidence")
    #ax.set_ylim([0,1])
    #save_plot(plt,output_name + "incorrect")

    fig, ax   = plt.subplots(figsize=(8, 6))
    ax.scatter(correct["char_length"], correct["confidence"], color='blue', alpha=0.02,s=4)
    ax.scatter(incorrect["char_length"], incorrect["confidence"], color='red', alpha=0.01,s=4)
    ax.set_xlabel("Question length")
    ax.set_ylabel("Confidence")
    ax.set_ylim([0,1])
    #ax.set_title(model)
    save_plot(plt,output_name + "correct_incorrect")
    
     
def save_predictions_plot(x:list[float], y:list[float], labels:tuple[str], output_name,colors=None):
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))

  
    ax.scatter(x, y, color='blue', alpha=0.1,s=4)

    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    save_plot(plt,labels,output_name)

    