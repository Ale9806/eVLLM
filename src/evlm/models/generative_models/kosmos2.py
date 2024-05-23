import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import sys
from pathlib import Path
from PIL import Image
FIXTURES_PATH = str(Path(__file__).resolve().parent)
FIXTURES_PATH = (Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class Kosmos2(BaseVLLM):
    def __init__(self,eval_mode:bool=True,context_length:int=None,verbose:bool=True,max_new_tokens:int=20):
        super().__init__(eval_mode,context_length,verbose,max_new_tokens)
        self.model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.preprocess = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.load_model()


    def forward(self,images:str, texts:str) -> dict[str,list[float]]:
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
            inputs  = self.preprocess(text="<grounding>" + texts, images=images, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=self.max_new_tokens,
            )
            generated_text  =  self.preprocess.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text, entities = self.preprocess.post_process_generation(generated_text)
            output["text"] = processed_text
            output["entities"] = entities

            self.handle_output(output,prompt=texts)
                
        return output


if __name__ == "__main__":
    model = Kosmos2()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "Describe the object in the picture"
    output = model.forward(images[0],prompt)
    print(output)
    import pdb;pdb.set_trace()

    
  

