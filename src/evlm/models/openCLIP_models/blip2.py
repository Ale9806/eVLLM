 
import torch
import sys
from pathlib import Path
from transformers import Blip2Processor, Blip2Model


FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class BLIP2(BaseCLIP):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.model      = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)   
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
        processed_images = self.preprocess_image(images, return_tensors ="pil")
        #import pdb;pdb.set_trace()

        with torch.no_grad():
            inputs  = self.preprocess(text=texts, images=processed_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits  = ooutputs.logits_per_image.softmax(dim=-1)
            output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]  = logits.to("cpu")
            output = self.handle_output(output,texts) 

        return output

if __name__ == "__main__":
    model = BLIP2()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    import pdb;pdb.set_trace()





 

 

    
  

    

 

    
  