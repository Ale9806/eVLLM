 
import torch
import sys
from pathlib import Path
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class BLIP2(BaseVLLM):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.model      = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")   
        self.preprocess = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.load_model()

    def forward(self,images:list[str], texts:list[str]) -> dict[str,list[float]]:
        """
        Forward pass through the model.

        Args:
            images: Input images 
            texts: Input texts.

        Returns:
            dict: Dictionary containing predictions and optionally class probabilities.

        
        """
        output:dict     = {}
        raw_image = Image.open(images).convert('RGB')
        with torch.no_grad():
            inputs  = self.preprocess(text=texts, images=raw_image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            generated_text = self.preprocess.decode(outputs[0], skip_special_tokens=True).strip()
            output["text"] = generated_text

        return output

if __name__ == "__main__":
    model = BLIP2()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "How many cats in the image?"
    output = model.forward(images[0],prompt)
    print(output)
    import pdb;pdb.set_trace()





 

 

    
  

    

 

    
  