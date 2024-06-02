import glob
import os
import json
from pathlib import Path
from random import shuffle

from torch.utils.data import Dataset,DataLoader

class JsonDataset(Dataset):
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

        if self.split:
            self.path: Path=  self.path / self.split

        self.files: list[str] = self.get_files()

    
        #import pdb;pdb.set_trace()
    def get_name():
        return self.name

    def get_files(self):
        files = glob.glob(str(self.path) + f"/*.{self.extension}")

        if self.shuffle_dataset:
            shuffle(files)

        if self.limit:
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
    split: str = 'test'
    DATA_ROOT = Path("/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/")
    sub_datasets: list[str] = os.listdir(DATA_ROOT)
    sub_datasets = list(set(sub_datasets) - set(['bravura.tex', 'bravura.md','image_json_pairs.jsonl', 'bravura.pdf', 'bravura.feather',"burgess_et_al_2024"]))
    sub_datasets = ["cognition"]
    dataset_path = DATA_ROOT / sub_datasets[0] 
    print(dataset_path)
    custom_dataset = JsonDataset(dataset_path,split=split, limit=1000)
    data_loader    = DataLoader(custom_dataset, batch_size=2, collate_fn=collate_fn)

    # Iterate over the data loader
    for data_point in custom_dataset:
        print(data_point)  # Here 'batch' will contain a list of JSON data from the files
        import pdb;pdb.set_trace()
        
