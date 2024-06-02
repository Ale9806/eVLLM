import torch
import sys
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
import random 

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)
from baseclip import BaseCLIP

random.seed(42)

class Random_model(BaseCLIP):
    """
    Random class for utilizing the BiomedCLIP model.

    Args:
        eval_mode (bool, s optional): Whether to set the model in evaluation mode. Defaults to True.
        context_length (int, optional): Length of input context. Defaults to 256.
    """
    random.seed(42)
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
        

    def forward(self,images:list[str], texts:list[str]) -> dict[str,list[float]]:
        """
        Forward pass through the model.

        Args:
            images: Input images 
            texts: Input texts.

        Returns:
            dict: Dictionary containing predictions and optionally class probabilities.

        
        """
        output = dict()
        #import pdb;pdb.set_trace()
        model_answers = texts.copy()
        random.shuffle(model_answers) 
        output["pred_prompt"] = model_answers
        output["pred"] = model_answers

        return output

if __name__ == "__main__":
    rm = Random_model()