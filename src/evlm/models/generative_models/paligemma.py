import torch
import requests
from transformers import AutoProcessor
from PIL import Image
import requests
import torch
from pathlib import Path
import sys
FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class PaliGemma(BaseVLLM):
    
    def __init__(self,model_id:str= "google/paligemma-3b-mix-224",
                      eval_mode:bool=True,
                      context_length:int= 1900,
                      verbose:bool=True,
                      max_new_tokens:int=100,
                      bfloat16:bool=True):

        super().__init__(eval_mode,context_length,verbose)
        access_token = "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"

        self.device     = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.processor  = AutoProcessor.from_pretrained(model_id,token=access_token)
        from transformers import PaliGemmaForConditionalGeneration

        if bfloat16:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    revision="bfloat16",
                    token=access_token)
        
        else:
            self.model      = PaliGemmaForConditionalGeneration.from_pretrained(model_id,token=access_token)
        self.max_new_tokens =max_new_tokens
        self.load_model()

    
    def forward(self,image:str, text:str) -> dict[str,str]:
        output:dict     = {}
        image  = Image.open(image).convert('RGB')
        inputs =  self.processor (text=text, images=image, return_tensors="pt").to(self.device)
        p_len  = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens , do_sample=False)
            generation = generation[0][p_len:]
            decoded    = self.processor .decode(generation, skip_special_tokens=True)
           

      
            output["text"] =  decoded

            return  output

if __name__ == "__main__":
    model = PaliGemma()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "segment animal"
    output:dict       = model.forward(images[0],prompt)
    print(output)
 