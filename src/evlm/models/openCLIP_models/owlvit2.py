import torch
import sys
from pathlib import Path
from transformers import Owlv2Processor, Owlv2ForObjectDetection

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class OwlVIT2(BaseCLIP):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.preprocess = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model  = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
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
        target_sizes      = torch.Tensor([ image.size[::-1] for image in processed_images]).to(self.device)
        with torch.no_grad():
            #import pdb;pdb.set_trace()
            inputs  = self.preprocess(text=texts, images=processed_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            results = self.preprocess.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            
            output["pred"]  = []
            output["probs"] = []
            output["bbox"]  = []
            for result in results:
                output["probs"].extend(result["scores"].to("cpu").tolist())
                output["pred"].extend(result["labels"].to("cpu").tolist())
                output["bbox"].append(result["boxes"].to("cpu").tolist())

            output = self.handle_output(output,texts) 
        #import pdb;pdb.set_trace()
        return output

if __name__ == "__main__":
    model = OwlVIT2()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    #import pdb;pdb.set_trace()





 

 

    
  