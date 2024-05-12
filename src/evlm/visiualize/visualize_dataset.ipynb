import sys
import inspect
import argparse
import importlib
from pathlib import Path
import matplotlib.pyplot as plt

module_path = str(Path(__file__).resolve().parent.parent) 
sys.path.append(module_path)

from torch.utils.data import DataLoader

from dataloaders.json_loader import JsonDataset

DATASETS = [
    'acevedo_et_al_2020', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', 'icpr2020_pollen',
    'nirschl_et_al_2018', 'jung_et_al_2022', 'wong_et_al_2022',
    'hussain_et_al_2019', 'colocalization_benchmark', 'kather_et_al_2016',
    'tang_et_al_2019', 'eulenberg_et_al_2017_brightfield',
    'burgess_et_al_2024_contour', 'nirschl_unpub_fluorescence',
    'burgess_et_al_2024_eccentricity', 'burgess_et_al_2024_texture',
    'held_et_al_2010']


def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a given model')
    parser.add_argument('--split',      type=str,  default='validation', help='Dataset split to evaluate (default is "validation")')
    parser.add_argument('--transform',  type=str,  default=None,         help='Data transformation function')
    parser.add_argument('--output_dir', type=str,  default='outputs',    help='Output directory to save evaluation results (default is "outputs")')
    parser.add_argument('--data_root',  type=str,  default='/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/', help='Dataset split to evaluate (default is "validation")')

    args = parser.parse_args()
    model_dict  = {}


    for datasets_ in DATASETS:

            dataset_dict              = {}
            dataset_dict["data_path"] = Path(args.data_root) /datasets_ 
            dataset_dict["dataset"] =  JsonDataset(dataset_path = dataset_dict["data_path"],split=args.split, limit=500)
            for data_point in dataset_dict["dataset"]: 
                import pdb;pdb.set_trace()
                data_point["metadata"]['name'] = dataset_dict["dataset"].path   / data_point["metadata"]['name']
                plt.imshow(data_point["metadata"]['name'] )
                
       


if __name__== "__main__":
    main()

## Example Usage:
# Enmbedding  (contrastive) Models:
## Base enviorment:
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model ALIGN
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BioMedCLIP
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model OpenCLIP
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model QuiltCLIP
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model PLIP
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BLIP

#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model OwlVIT2
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BLIP2

# Conch Enviorment:
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model ConchCLIP


## Base enviorment:
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model CogVLM
#python src/evlm/infernece/clip_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model QwenVLM
