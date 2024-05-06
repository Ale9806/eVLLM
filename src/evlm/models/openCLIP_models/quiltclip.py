import torch
import sys
from pathlib import Path
from open_clip import create_model_and_transforms, get_tokenizer

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP

class QuiltCLIP(BaseCLIP):
    """
    BioMedClip class for utilizing the BiomedCLIP model.

    Args:
        eval_mode (bool, s optional): Whether to set the model in evaluation mode. Defaults to True.
        context_length (int, optional): Length of input context. Defaults to 256.
    """

    def __init__(self,eval_mode:bool=True,context_length:int=77,verbose:bool=True):
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
        self.model, _, self.preprocess  = create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.tokenizer  = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
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
        output:dict       = {}
        processed_images = self.preprocess_image(images)
        tokenize_prompt  = self.tokenize(texts)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(processed_images, tokenize_prompt)
            logits:torch.Tensor  = (logit_scale * image_features @ text_features.T).detach().softmax(dim=-1)
            output["pred"]   = torch.argmax(logits, dim=1).to("cpu")
            output["probs"]  = logits.to("cpu")

            output = self.handle_output(output,texts) 

            return output


if __name__ == "__main__":
    model = QuiltCLIP()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompts:list[str] = ["An image of a cat","An image of a dog"]
    output:dict       = model.forward(images,prompts)
    print(output)
    #import pdb;pdb.set_trace()
