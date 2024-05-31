from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import re
from PIL import Image

import sys
from pathlib import Path

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from basevllm import BaseVLLM


class QwenVLM(BaseVLLM):

    def __init__(self,
                 eval_mode: bool = True,
                 context_length: int = 1900,
                 verbose: bool = True):
        super().__init__(eval_mode, context_length, verbose)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat",
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            device_map=self.device,
            trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.load_model()

    def forward(self, image: str, text: str) -> dict[str, str]:
        output: dict = {}
        query = self.tokenizer.from_list_format([{
            'image': str(image)
        }, {
            'text': text
        }])
        response = self.model.chat(self.tokenizer, query=query, history=None)
        output["text"], _ = response

        return output

    def forward_detect(self, image: str, class_name: str) -> dict[str, str]:
        """ do multi-instance, single class object detection """
        img = Image.open(image).convert('RGB')
        image_width, image_height = img.size

        prompt = f"Detect {class_name}"
        text_out = self.forward(image, prompt)['text']

        # interpret
        pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
        matches = re.findall(pattern, text_out)
        bboxes_ = [[int(item) / 1000 for item in sublist]
                   for sublist in matches]
        bboxes = [{
            'x1': m[0] * image_width,
            'y1': m[1] * image_height,
            'x2': m[2] * image_width,
            'y2': m[3] * image_height
        } for m in bboxes_]

        return bboxes


def test_detection():
    from PIL import Image, ImageDraw
    model = QwenVLM()

    img_path: str = "test_images"
    images: list[str] = [img_path + "/cat.jpeg", img_path + "/dog.jpeg"]
    images: list[str] = [img_path + "/dog.jpeg", img_path + "/cat.jpeg"]
    images = ["test_images/four_cups.png"]
    # images = ["test_images/four_coins.png"]
    # images: list[str] = [img_path + "/cat.jpeg", img_path + "/dog.jpeg"]

    # class_name = "cat"
    class_name = "cup"
    image = images[0]
    bboxes = model.forward_detect(image, class_name)

    # verify
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                       outline="red",
                       width=3)

    img.save("tmp.png")


if __name__ == "__main__":
    test_detection()
