import torch
import sys
from pathlib import Path
import conch
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer,tokenize

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class ConchCLIP(BaseCLIP):
    """
    BioMedClip class for utilizing the BiomedCLIP model.

    Args:

        eval_mode (bool, s optional): Whether to set the model in evaluation mode. Defaults to True.
        context_length (int, optional): Length of input context. Defaults to 256.
    """

    def __init__(self,eval_mode:bool=True,context_length:int=256,verbose:bool=True):
        """
        Initialize the Model object.

        Args:
            model: Pre-trained model for image-text matching.
            tokenizer: Tokenizer for processing captions.
            preprocess: Preprocessing function for images.
            context_length: Length of context for tokenization.
            device: Device to run the model on (e.g., "cuda" or "cpu").
        """
        super().__init__(eval_mode,context_length,verbose)
        self.model, self.preprocess  = create_model_from_pretrained('conch_ViT-B-16', "/pasteur/data/jnirschl/huggingface/hub/models--MahmoodLab--conch/snapshots/2e256beb1179fa0878ae97d50c646eb5db65966d/pytorch_model.bin")
        self.tokenizer = get_tokenizer()
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
        output:dict      = {}
        processed_images = self.preprocess_image(images)
        tokenize_prompts  =tokenize(texts=texts, tokenizer=self.tokenizer )  

        #import pdb;pdb.set_trace()
        with torch.no_grad():
            with torch.inference_mode():
                image_embedings  = self.model.encode_image(processed_images, proj_contrast=True, normalize=True).to(self.device)
                text_embedings   = self.model.encode_text(tokenize_prompts.to(self.device))
                
                logits = (image_embedings @ text_embedings.T).softmax(dim=-1)
                if len(logits.shape) ==1:
                    output["pred"]   = torch.argmax(logits).to("cpu")
                else:
                    output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
                    
                output["probs"]  = logits.to("cpu")

                output = self.handle_output(output,texts) 

                return output


if __name__ == "__main__":
    model = ConcCLIP()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    #import pdb;pdb.set_trace()
