from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

import sys
from pathlib import Path
FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class QwenVLM(BaseVLLM):
    def __init__(self,eval_mode:bool=True,context_length:int=1900,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.tokenizer               = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.model                   = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=self.device, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.load_model()

    def forward(self,image:str, text:str) -> dict[str,str]:
        output:dict     = {}
    
        query = [{'image': str(image)},{'text' : text}]

        query_tokens      = self.tokenizer.from_list_format(query)
        output["text"],_  = self.model.chat(self.tokenizer, query=query_tokens, history=None)
        return output
 

    

if __name__ == "__main__":
    model  = QwenVLM()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "Describe the object in the picture"


    response_example = model.forward(images[0],prompt)
    print(response_example)