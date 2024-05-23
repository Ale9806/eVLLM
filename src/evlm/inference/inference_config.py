import sys
from pathlib import Path
import importlib
module_path = str(Path(__file__).resolve().parent.parent) 
sys.path.append(module_path)
from  constants import  CLIP_MODELS,CHAT_MODELS, ALL_MODELS
import pandas as pd

def model_configuration(model_name:str) -> dict:
    """
    Configure a model based on the provided model name.

    Parameters:
        model_name (str): The name of the model to configure.

    Returns:
        dict: A dictionary containing the model's name, instance, and type.

    Raises:
        ValueError: If the provided model_name is not supported.

    Example:
        model_configuration("CLIP") returns:
        {'name': 'CLIP', 'model': <CLIPModel instance>, 'model_type': 'ENCODER'}
    """

    model_dict  = {}
    if model_name  in CLIP_MODELS:
        model_type = "ENCODER"
        module_path = f"models.openCLIP_models.{model_name.lower()}" # Construct the module path
        module_obj = importlib.import_module(module_path)            # Import the module dynamically
        
    elif model_name  in CHAT_MODELS:
        model_type = "GENERATIVE"
        module_path = f"models.generative_models.{model_name.lower()}" # Construct the module path
        module_obj = importlib.import_module(module_path)              # Import the module dynamically

    else:
        raise f"There is no support for model {model_name}"

    model_dict["name"]:str = model_name
    model_dict["model_type"]:str = model_type
    model_dict["model"] =  getattr(module_obj, model_name)()   
    

    return model_dict


def get_model_config(model_dict:dict) -> dict:
    import pdb; pdb.set_trace()
    
   
    #if model_dict["name"]  in ["ALIGN"]:
    #        text_endcoder_size:int   = sum(p.numel() for p in model_dict["model"].model.text_model.parameters())
    #        vision_endcoder_size:int = sum(p.numel() for p in model_dict["model"].model.vision_model.parameters())
    #        text_embed_dim:int       = model_dict["model"].model.text_embed_dim
    #        total_size:int           = text_endcoder_size + vision_endcoder_size

    #if model_dict["name"]  in ["BLIP"]:
    #        text_endcoder_size:int   = sum(p.numel() for p in model_dict["model"].model.text_encoder.parameters())
    #        vision_endcoder_size:int = sum(p.numel() for p in  vparameters())
   

    #elif  model_dict["name"]  in  ["OpenCLIP","PLIP","BioMedCLIP","ConchCLIP"]:
    #    text_endcoder_size:int   = sum(p.numel() for p in model_dict["model"].model.visual.parameters())
    #    vision_endcoder_size:int = sum(p.numel() for p in model_dict["model"].transformer.parameters())
       

   # elif model_dict["name"]  in CHAT_MODELS:
    #    text_endcoder_size:int   = 0
    #    vision_endcoder_size:int = 0
    #    total_size:int           = sum(p.numel() for p in model_dict["model"].model.parameters())
   
    #else:
    #    text_endcoder_size:str =  0 
    #    vision_endcoder_size:str = 0 
    #    total_size:str = 0

    total_size:int           = sum(p.numel() for p in model_dict["model"].model.parameters())

    
    ##text_endcoder_size:str = f"{text_endcoder_size:,}"
    #vision_endcoder_size:str = f"{vision_endcoder_size:,}"
    total_size:str = f"{total_size:,}"

    #return {"total_params":total_size,"text_endcoder_params":text_endcoder_size,"vision_encoder_params":vision_endcoder_size,"text_embed_dim":text_embed_dim}
    return {"model_name": model_dict["name"], "total_params": total_size}

if __name__ == "__main__":
    configs = []
    ALL_MODELS = ["CLIP"]
    for i,model_name in enumerate(ALL_MODELS):
        try:
            model_dict:dict = model_configuration(model_name) 
            config_ = get_model_config(model_dict)
            configs.append(config_)
            del model_dict
            gc.collect()
            torch.cuda.empty_cache() 
        except:
            print(f"Could not parse:{model_name}")
      

    df = pd.DataFrame(configs)
    df.to_csv("outputs/model_configs.csv", index=False)

    