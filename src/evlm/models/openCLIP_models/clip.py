import torch
import sys
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class CLIP(BaseCLIP):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        processed_images = self.load_image(images)

        with torch.no_grad():
            inputs = self.preprocess(text=texts, images=processed_images, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits  =  outputs.logits_per_image 
            probs   = logits.softmax(dim=1)
            output["pred"]:torch.Tensor   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]:torch.Tensor  = probs.to("cpu")
            output:dict[str,list[float]]  = self.handle_output(output,texts) 
            return output

if __name__ == "__main__":
    model = CLIP()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images[0],prompts)
    print(output)
    import pdb;pdb.set_trace()
