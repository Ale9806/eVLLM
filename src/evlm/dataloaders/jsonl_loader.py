import glob
import os
import json
from pathlib import Path
from random import shuffle

from torch.utils.data import Dataset,DataLoader

def check_json_file(file_path):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            return
        
        if data:  # Checks if the data is not empty
            return True

        elif data == {}:
            return False

        else:
            return False
            
def read_jsonl(file_path):
    """
    Read data from a JSONL (JSON Lines) file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: List of dictionaries, each containing data from one line of the JSONL file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    return data

class JsonlDataset(Dataset):
    """
    Dataset class to load JSON files.

    Args:
        dataset_path (str | Path): Path to the dataset directory.
        extension (str, optional): File extension of the JSON files. Defaults to "json".
        limit (int, optional): Maximum number of files to load. Defaults to None.
        shuffle_dataset (bool, optional): Whether to shuffle the dataset. Defaults to True.
    """

    def __init__(self, dataset_path: str | Path, 
                       split: str = None,
                       extension: str = "json", 
                       limit: int = None, 
                       shuffle_dataset: bool = False,
                       verbose:bool=True):

        
        if isinstance(dataset_path,str):
            dataset_path: Pat = Path(dataset_path)
            
    
        self.path: Path     = dataset_path
        self.name:str       =  dataset_path.name
        self.split:str      = split
        self.extension: str = extension
        self.limit: int = limit
        self.shuffle_dataset: bool = shuffle_dataset
        self.verbose:bool = verbose
        self.files: list[str] = self.get_files()

    
        #import pdb;pdb.set_trace()
    def get_name():
        return self.name

    def get_files(self):
        data_path = os.path.join(self.path,"test_200.jsonl")
        data  = read_jsonl(data_path)
        files_raw = [ os.path.join(self.path,data_point['json_file']) for data_point in data]
        files = [ file for file in files_raw if check_json_file(file)]
        files_lost = len(files_raw) -len(files)

        print(f"files lost {files_lost}")

        #xsimport pdb;pdb.set_trace()
        if self.shuffle_dataset:
            shuffle(files)

        if self.limit != None:
            files = files[:self.limit]

        
        print("="*80)
        print(f"Dataset {self.path} initilized with \n {len(files)}  datapoints")
        print("="*80)

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            data = json.load(f)
        return data

def collate_fn(batch):
    """
    Custom collate function to handle variable length data in the batch.
    """
    return batch


if __name__ == "__main__":
    split: str = 'validation'
    DATA_ROOT = Path("/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/")
    sub_datasets: list[str] = os.listdir(DATA_ROOT)
    sub_datasets = list(set(sub_datasets) - set(['bravura.tex', 'bravura.md','image_json_pairs.jsonl', 'bravura.pdf', 'bravura.feather',"burgess_et_al_2024"]))

    dataset_path = DATA_ROOT / sub_datasets[0] 
    print(dataset_path)
    custom_dataset = JsonlDataset(dataset_path,split=split, limit=1000)
    data_loader    = DataLoader(custom_dataset, batch_size=2, collate_fn=collate_fn)

    # Iterate over the data loader
    for data_point in custom_dataset:
        print(data_point)  # Here 'batch' will contain a list of JSON data from the files
        import pdb;pdb.set_trace()
        
