 
import torch
import sys
from pathlib import Path
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class InstructBLIP(BaseCLIP):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.model      = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.preprocess = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

       
    def forward(self,images:str, prompt:str) -> dict[str,list[float]]:
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
            inputs  = self.preprocess(text=prompt, images=processed_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(
                                            **inputs,
                                            do_sample=False,
                                            num_beams=5,
                                            max_length=256,
                                            min_length=1,
                                            top_p=0.9,
                                            repetition_penalty=1.5,
                                            length_penalty=1.0,
                                            temperature=1,
                                        )
            import pdb;pdb.set_trace()
            logits  = ooutputs.logits_per_image.softmax(dim=-1)
            output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]  = logits.to("cpu")
            output = self.handle_output(output,texts) 

        return output

if __name__ == "__main__":
    model = InstructBLIP()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "Describe the object in the picture"
    
    output:dict       = model.forward(images,prompt)
    print(output)
    import pdb;pdb.set_trace()





 

 

    
  

    

 

    
  