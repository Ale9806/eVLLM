""" This is an argparse wrapper for clip_inference.py, to edit behaviore or see code please go to functions at clip_inference.py"""
import sys
import logging
import inspect
import argparse
import importlib
from pathlib import Path
from torch.utils.data import DataLoader
import gc   
import torch
module_path = str(Path(__file__).resolve().parent.parent) 
sys.path.append(module_path)

from dataloaders.json_loader import JsonDataset,collate_fn
from dataloaders.jsonl_loader import JsonlDataset
from inference_config import model_configuration
import clip_inference
import generative_inference
from  constants import  DATASETS, CLIP_MODELS, CHAT_MODELS,QUESTIONS,ALL_MODELS

def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a given model')
    parser.add_argument('--dataset_name', type=str,                     help='Name of the dataset to evaluate')
    parser.add_argument('--model',        type=str,                     help='Name of model to evaluate')
    parser.add_argument('--split',        type=str,  default='test',    help='Dataset split to evaluate (default is "test")')
    parser.add_argument('--limit',        type=str,  default=None,      help='Limit of data points to evaluate (None evaluates everything )')
    parser.add_argument('--transform',    type=str,  default=None,      help='Data transformation function')
    parser.add_argument('--output_dir',   type=str,  default='outputs', help='Output directory to save evaluation results (default is "outputs")')
    parser.add_argument('--DEBUG',        action='store_true',          help='Flag to enable debug mode (only run inference on first  2 points)')
    parser.add_argument('--data_root',    type=str,  default='/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/', help='Dataset split to evaluate (default is "validation")')
    
    args:dict = parser.parse_args()

    if args.model  == "all":
        print("Running Inference on all Models")
        model_list:list[str] = ALL_MODELS

    elif args.model  == "gen":
        print("Running Inference on all generative Models")
        model_list:list[str] = CHAT_MODELS
    
    elif args.model  == "rep":
        print("Running Inference on all representation Models")
        model_list:list[str] = CLIP_MODELS
        
    else:
        model_list:list[str]  = [args.model]

    for model_name in model_list:
        model_dict:dict = model_configuration(model_name) 

        logger = logging.getLogger(__name__)
        output_name = Path(args.output_dir)/  f"{model_dict['name']}_logger_error.log"
        logging.basicConfig(filename=output_name, level=logging.INFO)

        if args.dataset_name  == "all":
            print("Running Infrence on all Datasets")
            dataset_list:list[str] = DATASETS
        else:
            dataset_list:list[str]  = [args.dataset_name]

        for datasets_ in dataset_list:
            dataset_dict = {}
            dataset_dict["data_path"] = Path(args.data_root) /datasets_ 
            if datasets_ == "cognition":
                dataset_dict["dataset"] =  JsonDataset(dataset_path = dataset_dict["data_path"],split=args.split, limit=args.limit)
            else:
                dataset_dict["dataset"] =  JsonlDataset(dataset_path = dataset_dict["data_path"],split=args.split, limit=args.limit)
            dataset_dict["loader"]  =  dataset_dict["dataset"] #DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
            #import pdb; pdb.set_trace()
            output_file:str       = dataset_dict["dataset"].name + ".csv"
            output_dir_name:Path = Path(args.output_dir) / model_dict['name'] / output_file
            #if output_dir_name.is_file():
            #    print(f"results {output_file} have already been generated")
            #else:
            print(f"Running Inference for {output_file}")
            try:
                do_inference(dataset_dict,model_dict,args,logger=logger)
            except:
                print(f"Could not do infernece for {model_name}:{datasets_}")




def do_inference(
    dataset_dict:dict,
    model_dict:dict,
    args:dict,
    logger=None) -> None:

    try:
        if model_dict["model_type"] == "ENCODER":
            clip_inference.evaluate_dataset(
                dataset = dataset_dict, 
                model_dict = model_dict, 
                split = args.split, 
                transform = args.transform, 
                output_dir = Path(args.output_dir), 
                DEBUG = args.DEBUG)

        elif model_dict["model_type"] == "GENERATIVE":
            generative_inference.evaluate_dataset(
                dataset = dataset_dict, 
                model_dict = model_dict, 
                split = args.split, 
                transform = args.transform, 
                output_dir = Path(args.output_dir),
                question_key  = "questions",
                DEBUG = args.DEBUG)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Could not evaluate {dataset_dict['data_path']}")
        if logger:
            logger.info(f"Could not evaluate {dataset_dict['data_path']}")
            logger.info(f"An error occurred: {e}")
    
    
    ### delet model f
    del model_dict
    gc.collect()
    torch.cuda.empty_cache() 
    

if __name__ == "__main__":
    main()

## Example Usage:
# Enmbedding  (contrastive) Models:
## Base enviorment:
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model all
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BioMedCLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model OpenCLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model QuiltCLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model PLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BLIP

#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model OwlVIT2
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BLIP2

# Conch Enviorment:
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model ConchCLIP

#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model BioMedCLIP

#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model ConchCLIP


## Base enviorment:
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model CogVLM
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model QwenVLM
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model gen

#python src/evlm/inference/model_inference_wrapper.py --dataset_name "acevedo_et_al_2020" --model Random_model
#python src/evlm/inference/model_inference_wrapper.py --dataset_name "colocalization_benchmark" --model CogVLM

#python src/evlm/inference/model_inference_wrapper.py --dataset_name cognition_0601 --model ALIGN