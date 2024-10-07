import torch
import sys
from pathlib import Path
from open_clip import create_model_and_transforms, get_tokenizer

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class CoCa(BaseCLIP):
    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
        #'laion2b_s34b_b79k'
        super().__init__(eval_mode,context_length,verbose)
        self.model, _, transform = create_model_and_transforms(model_name='coca_ViT-B-32',pretrained='laion2b_s13b_b90k')
        self.tokenizer = get_tokenizer('coca_ViT-B-32')
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
        #processed_images = self.preprocess_image(images)
        tokenize_prompt  = self.tokenize(texts)

        with torch.no_grad():
            text_features  = self.model.encode_text(tokenize_prompt)
            import pdb;pdb.set_trace()
            image_features = self.model.encode_image(processed_images)
            text_features  = self.model.encode_text(tokenize_prompt)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /=  text_features.norm(dim=-1, keepdim=True)
            logits         = (image_features @ text_features.T).softmax(dim=-1)

            output["pred"]:torch.Tensor    = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]:torch.Tensor  = logits.to("cpu")
            output:dict[str,list[float]]  = self.handle_output(output,texts) 

            return output

if __name__ == "__main__":
    model = CoCa()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    import pdb;pdb.set_trace()
