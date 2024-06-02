import torch
import requests
from transformers import AutoProcessor
from PIL import Image
import requests
import torch
from pathlib import Path
import re
import sys

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM


class PaliGemma(BaseVLLM):

    def __init__(self,
                 model_id: str = "google/paligemma-3b-mix-224",
                 eval_mode: bool = True,
                 context_length: int = 1900,
                 verbose: bool = True,
                 max_new_tokens: int = 100,
                 bfloat16: bool = True):

        super().__init__(eval_mode, context_length, verbose)
        access_token = "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"

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.processor = AutoProcessor.from_pretrained(model_id,
                                                       token=access_token)
        from transformers import PaliGemmaForConditionalGeneration

        if bfloat16:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                revision="bfloat16",
                token=access_token)

        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, token=access_token)
        self.max_new_tokens = max_new_tokens
        self.load_model()

    def forward(self, image: str, text: str) -> dict[str, str]:
        output: dict = {}
        image = Image.open(image).convert('RGB')
        inputs = self.processor(text=text, images=image,
                                return_tensors="pt").to(self.device)
        p_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True)
        generation = generation_output.sequences[0][p_len:]
        output["text"] = self.processor.decode(generation,
                                               skip_special_tokens=True)

        # compute probs for A,B,C,D,E,F for the first token
        first_token_logits = generation_output.scores[0][0]
        first_token_probs = torch.softmax(first_token_logits, dim=0)
        choices = ["A", "B", "C", "D", "E", "F"]
        ids_choices = [
            self.processor.tokenizer.encode(s, add_special_tokens=False)[0]
            for s in choices
        ]
        output['probs_choices'] = dict(
            zip(choices, first_token_probs[ids_choices].cpu().numpy()))

        # the 'confidence' is the prob for the first char. if it was not in `choices`, return nan.
        try:
            output['confidence'] = output['probs_choices'].get(
                output['text'][0], torch.nan)
        # incase len(output['text'])==0
        except IndexError:
            output['confidence'] = torch.nan

        return output

    def forward_detect(self, image: str, class_name: str) -> dict[str, str]:
        """
        Run detection, and if it returns nothing, run segmentation, and get
        the bbox. 

        Returns: List[dict] each list element is one detection, where the keys 
            are the bbox rectangle coords: 'x1','x2','y1','y2'. The coordinates 
            are in the image space without any resizing.
        """

        def forward_generation(text, img):
            """ the basic forward pass, since we may run it twice """
            inputs = self.processor(text=text, images=img,
                                    return_tensors="pt").to(self.device)
            p_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True)
            generation = generation_output.sequences[0][p_len:]
            text_out = self.processor.decode(generation,
                                             skip_special_tokens=True)

            return text_out

        output: dict = {}
        img = Image.open(image).convert('RGB')
        image_width, image_height = img.size

        # run detection
        text = f"detect {class_name}"
        text_out = forward_generation(text, img)

        # there are images where detection fails, but segmentation works. 
        # so if detection fails, then get the bbox from segmentation result
        if text_out == "":
            text = f"segment {class_name}; {class_name}"
            text_out = forward_generation(text, img)

        # recover bbox coordinate objects
        bboxes_ = [re.findall(r'<loc(\d{4})>', t) for t in text_out.split(";")]
        bboxes_ = [[int(item) for item in sublist] for sublist in bboxes_]

        # put them in the image coords
        bboxes = []
        for bbox_ in bboxes_:
            # missing
            if len(bbox_) == 0:
                continue

            # text response can give the wrong number of coords
            elif len(bbox_) > 4:
                bbox = bbox_[:4]

            # regular case
            else:
                bbox = bbox_

            # paligemma coord system uses 1024
            y_min, x_min, y_max, x_max = [n / 1024 for n in bbox]
            bbox_dict = {
                "x1": x_min * image_width,
                "x2": x_max * image_width,
                "y1": y_min * image_height,
                "y2": y_max * image_height,
            }

            bboxes.append(bbox_dict)

        output['bboxes'] = bboxes

        return output

def test_detection():
    from PIL import Image, ImageDraw
    model = PaliGemma()
    img_path: str = "test_images"
    images: list[str] = [img_path + "/cat.jpeg", img_path + "/dog.jpeg"]

    images: list[str] = [img_path + "/dog.jpeg", img_path + "/cat.jpeg"]
    images = ["test_images/four_cups.png"]
    images = ["test_images/four_coins.png"]

    prompt: str = "segment animal"
    prompt: str = "detect cup"
    prompt: str = "detect coin"

    output = model.forward_detect(images[0], prompt)
    bboxes = output['bboxes']

    img = Image.open(images[0]).convert("RGB")
    image_width, image_height = img.size
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                       outline="red",
                       width=3)

    img.save("tmp.png")


if __name__ == "__main__":
    test_detection()