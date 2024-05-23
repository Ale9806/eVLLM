from abc import ABC,abstractmethod
import torch 

class BaseVLLM(ABC):
    """
    BaseVLLM class for utilizing the models based on OpenCLIP.

    Args:
        eval_mode (bool, optional): Whether to set the model in evaluation mode. Defaults to True.
        context_length (int, optional): Length of input context. Defaults to 256.
    """

    def __init__(self,eval_mode:bool=True,context_length:int=256,verbose:bool=True,max_new_tokens:int=20):
        """
        Initialize the Model object.

        Args:
            model: Pre-trained model for image-text matching.
            tokenizer: Tokenizer for processing captions.
            preprocess: Preprocessing function for images.
            context_length: Length of context for tokenization.
            device: Device to run the model on (e.g., "cuda" or "cpu").
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.context_length:int = context_length
        self.eval:bool = eval_mode
        self.verbose:bool = verbose
        self.max_new_tokens:int = max_new_tokens

        if self.verbose:
          print()
          print("="*80)
          print(f"{self.__class__.__name__} model initalized with Eval:{self.eval}, Context Length:{self.context_length}, and device:{self.device}")
          print("="*80)
          print()


    def load_model(self) -> None:
        """
        Loads the model onto the specified device and sets it to evaluation mode if `eval` flag is set.

        Returns:
            None
        """
        self.model.to(self.device)

        if self.eval:
            self.model.eval()
      

    def tokenize(self,captions:list[str]) -> torch.Tensor:
        """
        Tokenize captions using the biomedclip's tokenizer (self.tokenizer).

        Args:
            captions: Captions to tokenize, either a string or a list of strings.

        Returns:
            Tensor: Tokenized captions.
        """
        return self.tokenizer(captions, context_length=self.context_length).to(self.device)


    def preprocess_image(self,image: str | list[str],return_tensors:str="pt") -> torch.Tensor:
        """
        Preprocess images before feeding them into the model.

        Args:
            image: Image file path or a list of image file paths.

        Returns:
            Tensor: Preprocessed images.
        """
        if return_tensors == "pt":
            if isinstance(image, str) or isinstance(image, pathlib.Path):
                return torch.stack([self.preprocess(Image.open(image))]).to(self.device)

            elif isinstance(image, list) or isinstance(image, tuple):
                return torch.stack([self.preprocess(self.load_image(img)) for img in image]).to(self.device)

        elif return_tensors == "pil":
            if isinstance(image, str) or isinstance(image, pathlib.Path):
                return [Image.open(image)]

            elif isinstance(image, list) or isinstance(image, tuple):
                return [self.load_image(img) for img in image]


    def load_image(self,image):
        """
        Loads an image from file path or PIL Image object.

        Args:
            image: A file path (string or pathlib.Path) or a PIL Image object.

        Returns:
            PIL Image object: The loaded image.
        """
        if isinstance(image, str) or isinstance(image, pathlib.Path):
            image = Image.open(image)
        return image

    def handle_output(self,output:dict[str,torch.Tensor],prompt:str) -> dict[str,list[int | str]]:
        """
        Process the output dictionary.

        Parameters:
        - output (dict[str, torch.Tensor]): Dictionary containing tensors.
        - prompts (List[str], optional): List of prompts. Defaults to None.

        Returns:
        - dict[str, List[Union[int, str]]]: Processed output dictionary.

        Note:
        - The function converts tensor values in the output dictionary to lists.
        - If prompts are provided, it adds a "pred_prompt" key to the output, which maps indices from the "pred" key to the corresponding prompts.
        """
        prompt_length = len(prompt)
        if output["text"].startswith(prompt):
            output["text"] = output["text"][prompt_length:].strip()
        return output
            
    @abstractmethod
    def forward(self,images, texts, return_probs:bool=True) -> dict[str,torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images.
            texts: Input texts.
            return_probs: Whether to return class probabilities along with predictions.

        Returns:
            dict: Dictionary containing predictions and optionally class probabilities.

        """
        pass
      
