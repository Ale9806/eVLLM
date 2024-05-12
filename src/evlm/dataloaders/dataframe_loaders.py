import glob
import os
import json
from pathlib import Path
from random import shuffle

from torch.utils.data import Dataset,DataLoader

import numpy as np
import pandas as pd




def create_dataframe(
    data_root: str, 
    split_root: str,
    path_name:str='Path') -> pd.DataFrame:
    
    if isinstance(data_root, str):
        data_root = Path(data_root)

    image_path = data_root / split_root
    image_files = list(Path(image_path).glob("patient*/study*/*.jpg"))

    df_path = str(image_path) + ".csv"
    df = pd.read_csv(df_path)

    df[path_name] = image_files

    return df


def process_data(
    df: pd.DataFrame, 
    class_names: list[str] = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'],
    path_name: str = 'Path', 
    convert_class_to_str: bool = True) -> tuple[list[str], list[list[str]]]:
    """
    Process DataFrame to extract images and corresponding classes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing image paths and class labels.
    class_names : list[str], optional
        List of class names, defaults to ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'].
    path_name : str, optional
        Name of the column containing image paths, defaults to 'Path'.
    convert_class_to_str : bool, optional
        Whether to convert class labels to class names, defaults to True.

    Returns:
    --------
    images : list[str]
        List of image paths.
    classes : list[list[str]]
        List of class labels corresponding to each image path.
    """

    images, classes = [], []

    for _, row in df.iterrows():
        image_path = str(row[path_name])
        class_labels = row[class_names].tolist()

        if convert_class_to_str:
            class_labels = [class_names[i] for i, val in enumerate(class_labels) if val == 1.0]

        images.append(image_path)
        classes.append(class_labels)

    return images, classes


class PandasDataframe(Dataset):
    """
    Dataset class to load Dataset from PandasDataframe fi

    Args:
        data_frame (pd.DataFrame): Dataframe dataset
        extension (str, optional): File extension of the JSON files. Defaults to "json".
        limit (int, optional): Maximum number of files to load. Defaults to None.
        shuffle_dataset (bool, optional): Whether to shuffle the dataset. Defaults to True.
    """

    def __init__(self, 
                 data_frame:pd.DataFrame,
                 class_names: list[str],
                 path_name:str, 
                 dataset_name:str="Chexpert",
                 convert_class_to_str: bool = True
                ):
        
        self.df = data_frame
        self.class_names = class_names
        self.name = dataset_name
        self.images, self.classes  = process_data(
            df=  self.df,
            class_names = class_names,
            path_name = path_name,
            convert_class_to_str =convert_class_to_str)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image:str = self.images[idx]
        classes:list[str] = self.classes[idx]
        return image,classes


def collate_fn(batch):
    """
    Custom collate function to handle variable length data in the batch.
    """
    return batch


if __name__ == "__main__":
    class_names:list[str] = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    path_name:str         ='Path'

    data_root:str  = "/pasteur/data/chexpert_small/CheXpert-v1.0-small"
    split_root:str = "valid"
    
    df_dataSet    = PandasDataframe(df,class_names,path_name,convert_class_to_str=True)
    df_dataLoader = DataLoader(df_dataSet, batch_size=16, shuffle=True, collate_fn=collate_fn)


