import torch
import sys
from pathlib import Path
from transformers import AutoProcessor, BlipModel
from transformers import BlipProcessor, BlipForImageTextRetrieval

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class BLIP(BaseCLIP):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        #self.model      =  BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        #self.preprocess = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model      =  BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco") 
        self.preprocess = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
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

        with torch.no_grad():
            inputs  = self.preprocess(text=texts, images=processed_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits  = outputs["itm_score"].softmax(dim=-1)
            output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]  = logits.to("cpu")
            output = self.handle_output(output,texts) 

        return output

if __name__ == "__main__":
    model = BLIP()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    #import pdb;pdb.set_trace()





 

 

    
  