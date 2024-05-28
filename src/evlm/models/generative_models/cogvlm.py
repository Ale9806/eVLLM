import torch
import requests
from PIL import Image
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, LlamaTokenizer

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM

class CogVLM(BaseVLLM):
    
    def __init__(self,eval_mode:bool=True,context_length:int= 1900,verbose:bool=True):
        super().__init__(eval_mode,context_length,verbose)
        self.device     = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer  = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model      = AutoModelForCausalLM.from_pretrained('THUDM/cogvlm-chat-hf',
                                                                torch_dtype=torch.bfloat16,
                                                                low_cpu_mem_usage=True,
                                                                trust_remote_code=True)
        self.load_model()

    
    def forward(self,image:str, text:str) -> dict[str,str]:
        output:dict     = {}
        image  = Image.open(image).convert('RGB')
        inputs = self.model.build_conversation_input_ids(self.tokenizer, 
                                                         query=text, history=[], images=[image])

        inputs = {
            'input_ids':      inputs['input_ids'].unsqueeze(0).to(self.device ),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device ),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device ),
            'images': [[inputs['images'][0].to(self.device ).to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": self.context_length, "do_sample": False, 
                "return_dict_in_generate":True, "output_scores":True}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
            output["text"] =  self.tokenizer.decode(generated_ids[0])

            # compute probs that first token was 'A', 'B', ..., 'F'
            scores = outputs.scores
            assert generated_ids.shape[1] == len(outputs.scores)
            choices = ["A","B","C","D","E","F"]
            ids_choices = [self.tokenizer.encode(s, add_special_tokens=False)[0] for s in choices]
            probs = torch.softmax(scores[0][0], dim=0)
            probs_choices_ = [probs[id_].item() for id_ in ids_choices]
            output['probs_choices'] = dict(zip(choices, probs_choices_))
            
            # the 'confidence' is the prob for the first char. if it was not in `choices`, return nan.
            output['confidence'] = output['probs_choices'].get(output['text'][0], torch.nan)

        return output
            

if __name__ == "__main__":
    model = CogVLM()
    img_path:str      = "/pasteur/u/ale9806/Repositories/evlm/test_images/"
    images:list[str]  = [img_path +"/cat.jpeg",img_path +"/dog.jpeg"]
    prompt:str        = "Describe the object in the picture"
    output:dict       = model.forward(images[0],prompt)
    print(output)
 