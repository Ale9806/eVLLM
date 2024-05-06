""" This is an argparse wrapper for clip_inference.py, to edit behaviore or see code please go to functions at clip_inference.py"""
import sys
import inspect
import argparse
import importlib
from pathlib import Path


module_path = str(Path(__file__).resolve().parent.parent) 
sys.path.append(module_path)

from torch.utils.data import DataLoader

from dataloaders.json_loader import JsonDataset,collate_fn
import clip_inference
import generative_inference


def evaluate_dataset_wrapper(dataset_name, model_dict, split="validation", transform=None, output_dir=Path("outputs"), DEBUG=False):
    """
    Function to evaluate a dataset using a given model.

    Parameters:
    - dataset_name: Name of the dataset to evaluate.
    - data_root: Root directory where the dataset is located.
    - model_dict: Dictionary containing model information.
    - split: Dataset split to evaluate (default is "validation").
    - transform: Data transformation function (default is None).
    - output_dir: Output directory to save evaluation results (default is "outputs").
    - DEBUG: Flag to enable debug mode (default is False).
    """

    evaluate_dataset( dataset_name,
                      model_dict  = model_dict, 
                      split       = "validation",
                      transform   = transform,
                      output_dir  = Path("outputs"),
                      DEBUG       =  DEBUG) 

def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a given model')
    parser.add_argument('--dataset_name', type=str,                        help='Name of the dataset to evaluate')
    parser.add_argument('--model',        type=str,                        help='Name of model')
    parser.add_argument('--split',      type=str,  default='validation', help='Dataset split to evaluate (default is "validation")')
    parser.add_argument('--transform',  type=str,  default=None,         help='Data transformation function')
    parser.add_argument('--output_dir', type=str,  default='outputs',    help='Output directory to save evaluation results (default is "outputs")')
    parser.add_argument('--DEBUG',      action='store_true',             help='Flag to enable debug mode')
    parser.add_argument('--data_root',  type=str,  default='/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/', help='Dataset split to evaluate (default is "validation")')
    

    args = parser.parse_args()
    model_dict  = {}

    if args.model not in ["ALIGN","QuiltCLIP","OwlVIT2","OpenCLIP","BLIP2","BLIP","PLIP","BioMedCLIP","ConchCLIP","CogVLM","QwenVLM"]:
        raise f"There is no support for model {args.model}"



    if args.model  in ["ALIGN","QuiltCLIP","OwlVIT2","OpenCLIP","BLIP2","BLIP","PLIP","BioMedCLIP","ConchCLIP"]:
        model_type = "ENCODER"
        module_path = f"models.openCLIP_models.{args.model.lower()}" # Construct the module path
        module_obj = importlib.import_module(module_path)            # Import the module dynamically
        

    elif args.model  in ["CogVLM","QwenVLM"]:
        model_type = "GENERATIVE"
        module_path = f"models.generative_models.{args.model.lower()}" # Construct the module path
        module_obj = importlib.import_module(module_path)              # Import the module dynamically

  
       
    class_obj  = getattr(module_obj, args.model)                 # Get a reference to the class object
    model_dict["name"] = args.model
    model_dict["model"]= class_obj()

    output_dir = Path(args.output_dir)

    # Call evaluate_dataset function with parsed arguments
    DATASETS = [
    'acevedo_et_al_2020', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', 'icpr2020_pollen',
    'nirschl_et_al_2018', 'jung_et_al_2022', 'wong_et_al_2022',
    'hussain_et_al_2019', 'colocalization_benchmark', 'kather_et_al_2016',
    'tang_et_al_2019', 'eulenberg_et_al_2017_brightfield',
    'burgess_et_al_2024_contour', 'nirschl_unpub_fluorescence',
    'burgess_et_al_2024_eccentricity', 'burgess_et_al_2024_texture',
    'held_et_al_2010']

    for datasets_ in DATASETS:

        dataset_dict = {}
        #dataset_dict["data_root"] = args.data_root
        #dataset_dict["data_name"] =  datasets_ #"jung_et_al_2022" #args.data_name
        #dataset_dict["split"]     = args.split
        dataset_dict["data_path"] = Path(args.data_root) /datasets_ 
        #'/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/jung_et_al_2022/validation'
    
        dataset_dict["dataset"] =  JsonDataset(dataset_path = dataset_dict["data_path"],split=args.split, limit=100)
        dataset_dict["loader"]  = dataset_dict["dataset"]  # DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        try:
            if model_type == "ENCODER":
                clip_inference.evaluate_dataset(dataset_dict, model_dict, args.split, args.transform, output_dir, args.DEBUG)

            elif model_type == "GENERATIVE":
                generative_inference.evaluate_dataset(dataset_dict, model_dict, args.split, args.transform, output_dir, args.DEBUG)

        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Could not evaluate {datasets_}")

if __name__ == "__main__":
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
