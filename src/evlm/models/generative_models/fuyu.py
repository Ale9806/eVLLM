import torch
from transformers import FuyuProcessor, FuyuForCausalLM
import sys
from pathlib import Path
from PIL import Image
FIXTURES_PATH = str(Path(__file__).resolve().parent)
FIXTURES_PATH = (Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class Fuyu(BaseVLLM):
    def __init__(self,eval_mode:bool=True,context_length:int=None,verbose:bool=True,max_new_tokens:int=20):
        super().__init__(eval_mode,context_length,verbose,max_new_tokens)
        self.preprocess  = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        self.model       = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map="cuda")
        #self.load_model()


    def forward(self,images:list[str], texts:str) -> dict[str,list[float]]:
        """
        Forward pass through the model.

        Args:
            images: Input images 
            texts: Input texts.

        Returns:
            dict: Dictionary containing predictions and optionally class probabilities.

        """
        output:dict     = {}
        images = Image.open(images)
        with torch.no_grad():
            inputs  = self.preprocess(text=texts, images=images, return_tensors="pt").to(self.device)
            generated_ids   =  self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_text  =  self.preprocess.batch_decode(generated_ids[:, -self.max_new_tokens:], skip_special_tokens=True)[0]
            output["text"] = generated_text
            self.handle_output(output,prompt=texts)
                
        return output

if __name__ == "__main__":
    model = Fuyu()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "Describe the object in the picture"
    output = model.forward(images[0],prompt)
    print(output)
    import pdb;pdb.set_trace()

    
  

