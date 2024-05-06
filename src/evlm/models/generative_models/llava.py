#from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

#from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration

FIXTURES_PATH = (Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class Llava(BaseVLLM):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)

        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.preprocess = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        #self.preprocess = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        #self.model  = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
        #                                                                 torch_dtype=torch.float16,
        #                                                                 low_cpu_mem_usage=True)
     

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
            import pdb;pdb.set_trace()
            logits  = outputs.logits_per_image.softmax(dim=-1)
            output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]  = logits.to("cpu")
            output = self.handle_output(output,texts) 

        return output

if __name__ == "__main__":
    model = LLAVA()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    import pdb;pdb.set_trace()

    
  

