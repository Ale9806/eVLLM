import torch
import requests
from PIL import Image
import sys
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM




FIXTURES_PATH = (Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class QuiltLlava(BaseVLLM):
    
    def __init__(self,eval_mode:bool=True,context_length:int= 1900,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.device     = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.processor  =  AutoProcessor.from_pretrained("wisdomik/Quilt-Llava-v1.5-7b")
        self.model = AutoModelForCausalLM.from_pretrained("wisdomik/Quilt-Llava-v1.5-7b")
        self.load_model()

    
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
        processed_images = self.preprocess_image(images, return_tensors ="pil")

        with torch.no_grad():
            inputs  = self.preprocess(text=texts, images=processed_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])
            results     = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            import pdb;pdb.set_trace()
            logits  = outputs.logits_per_image.softmax(dim=-1)
            output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]  = logits.to("cpu")
            output = self.handle_output(output,texts) 

        return output

if __name__ == "__main__":
    model = QuiltLlava()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "Describe the object in the picture"
    output:dict       = model.forward(images[0],prompt)
    print(output)
    import pdb;pdb.set_trace()

    
  

