"""
API for calling gpt. 
Uses the environment variable OPENAI_API_KEY.
Models: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo

Also creates a cache autoamtically - see lmdb library calls below
"""
from openai import OpenAI, ChatCompletion
from pathlib import Path
import base64
from PIL import Image
import numpy as np
import os
import io
from typing import List, Optional
import lmdb
import json
import sys
import logging
import concurrent.futures
from threading import Lock
import time
import tqdm
import hashlib
import math

import sys

sys.path.insert(0, "..")
sys.path.insert(0, ".")


class GptApi():

    def __init__(self, top_logprobs=10, do_cache=True):
        """ 
        most params configurable in the call_gpt function 
        caching creates a folder in 'cache/cache_openai'
        """
        self.top_logprobs = top_logprobs
        self.client = OpenAI()

        self.do_cache = do_cache
        if self.do_cache:
            Path("cache").mkdir(exist_ok=True)
            self.cache_openai = lmdb.open("cache/cache_openai",
                                          map_size=int(1e11))
            self.cache_lock = Lock()

    def forward(self, image: str, text: str, do_confidences: bool = True, **kwargs) -> dict[str, str]:
        """ 
        image (str) is path to image 
        text (str) is prompt text 
        kwargs (dict) is anything in 
        """
        output = {}

        # get response
        imgs = [np.array(Image.open(image).convert("RGB"))]
        msg, response = self.call_gpt(text, imgs, **kwargs)

        # text response 
        output['text'] = msg

        # process confidence scores
        if do_confidences:
            output['confidence'] = response['confidence']
            output['probs_choices'] = response['probs_choices']

        # compute cost, which is 0 if cached
        if 'prompt_tokens' in response.keys():
            prompt_tokens = response['prompt_tokens']
            completion_tokens = response['completion_tokens']
            output['cost'] = self.compute_api_call_cost(
                prompt_tokens, completion_tokens,
                kwargs.get("model", "gpt-4o"))
        else:
            output['cost'] = 0  # bc it was cached

        return output


    def call_gpt(
        self,
        # args for setting the `messages` param
        text: str,
        imgs: List[np.ndarray] = None,
        system_prompt: str = None,
        json_mode: bool = False,
        # kwargs for client.chat.completions.create
        detail: str = "high",
        model: str = "gpt-4o",
        temperature: float = 1,
        max_tokens: int = 2048,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        seed: int = 0,
        n: int = 1,
        # logprobs
        logprobs: bool = True,
        # args for caching behaviour
        cache: bool = True,
        overwrite_cache: bool = False,
        num_retries:
        # if json_mode=True, and not json decodable, retry this many time
        int = 3):
        """ 
        Call GPT LLM or VLM synchronously with caching.
        To call this in a batch efficiently, see func `call_gpt_batch`.

        If `cache=True`, then look in database ./cache/cache_openai for these exact
        calling args/kwargs. The caching only saves the first return message, and not
        the whole response object. 

        imgs: optionally add images. Must be a sequence of numpy arrays. 
        overwrite_cache (bool): do NOT get response from cache but DO save it to cache.
        seed (int): doesnt actually work with openai API atm, but it is in the 
            cache key, so changing it will force the API to be called again
        """
        if not self.do_cache:
            cache = False

        # response format
        if json_mode:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        # system prompt
        messages = [{
            "role": "system",
            "content": system_prompt,
        }] if system_prompt is not None else []

        # text prompt
        content = [
            {
                "type": "text",
                "text": text,
            },
        ]

        # for imgs, put a hash key representation in content for now. If not cahcing,
        # we'll replace this value later (it's because `_encode_image_np` is slow)
        if imgs:
            content.append({"imgs_hash_key": [hash_array(im) for im in imgs]})

        # text & imgs to message - assume one message only
        messages.append({"role": "user", "content": content})

        # arguments to the call for client.chat.completions.create
        kwargs = dict(
            messages=messages,
            response_format=response_format,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            n=n,
        )
        if logprobs:
            kwargs['logprobs'] = logprobs
            kwargs['top_logprobs'] = self.top_logprobs

        if cache:
            cache_key = json.dumps(kwargs, sort_keys=True)
            with self.cache_lock:
                cached_value = get_from_cache(cache_key, self.cache_openai)
            if cached_value is not None and not overwrite_cache:
                cached_value = json.loads(cached_value)
                msg = cached_value['msg']
                response_out = cached_value['response_out']
                if json_mode:
                    msg = json.loads(msg)
                return msg, response_out

        # not caching, so if imgs,then encode the image to the http payload
        if imgs:
            assert "imgs_hash_key" in content[-1].keys()
            content.pop()

            base64_imgs = [_encode_image_np(im) for im in imgs]
            for base64_img in base64_imgs:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}",
                        "detail": detail,
                    },
                })

        # call gpt if not cached. If json_mode=True, check that it's json and retry if not
        for i in range(num_retries):

            response = self.client.chat.completions.create(**kwargs)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            msg = response.choices[0].message.content

            if json_mode:
                try:
                    _ = json.loads(msg)
                    break  # successfully output json

                except json.decoder.JSONDecodeError:
                    if i == num_retries - 1:
                        raise f"Response not valid json, after {num_retries} tries"
                    logging.info(f"Response not valid json, retrying")
                    continue

        response_out = {}

        if logprobs:
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            response_out['confidence'] = math.exp(top_logprobs[0].logprob)
            response_out['probs_choices'] = dict(
                zip([t.token for t in top_logprobs],
                    [math.exp(t.logprob) for t in top_logprobs]))

        # save to cache if enabled
        if cache:
            with self.cache_lock:
                cached_value = json.dumps({
                    'msg': msg,
                    'response_out': response_out
                })
                save_to_cache(cache_key, cached_value, self.cache_openai)

        response_out['prompt_tokens'] = prompt_tokens
        response_out['completion_tokens'] = completion_tokens

        if json_mode:
            msg = json.loads(msg)

        return msg, response_out

    def forward_batch(self,
                      texts,
                      imgs=None,
                      seeds=None,
                      json_modes=None,
                      get_meta=True,
                      **kwargs):
        """ 
        with multithreading
        if return_meta, then return a dict that tells you the runtime, the cost
        """
        n = len(texts)
        if imgs is None:
            imgs = [None] * n

        assert n == len(imgs), "texts and imgs must have the same length"

        # handle having a different seed per call
        all_kwargs = [kwargs.copy() for _ in range(n)]
        if seeds is not None or json_modes is not None:
            for i in range(n):
                if seeds is not None:
                    all_kwargs[i]['seed'] = seeds[i]
                if json_modes is not None:
                    all_kwargs[i]['json_mode'] = json_modes[i]

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            futures = []

            for text, img, _kwargs in zip(texts, imgs, all_kwargs):
                future = executor.submit(self.forward, text, img, **_kwargs)
                futures.append(future)

            # run
            results = [list(future.result()) for future in futures]

        if get_meta:
            for i, (msg, tokens) in enumerate(results):

                if tokens is not None:
                    price = compute_api_call_cost(
                        tokens['prompt_tokens'], tokens['completion_tokens'],
                        kwargs.get("model", "gpt-4-turbo-2024-04-09"))
                else:
                    price = 0

                results[i][1] = price

        return results

    def compute_api_call_cost(self,
                              prompt_tokens: int,
                              completion_tokens: int,
                              model="gpt-4o"):
        """
        Warning: prices need to be manually updated from
        https://openai.com/api/pricing/
        """
        prices_per_million_input = {
            "gpt-4o": 5,
            "gpt-4-turbo": 10,
            "gpt-4": 30,
            "gpt-3.5-turbo": 0.5
        }
        prices_per_million_output = {
            "gpt-4o": 15,
            "gpt-4-turbo": 30,
            "gpt-4": 60,
            "gpt-3.5-turbo": 1.5
        }
        if "gpt-4o" in model:
            key = "gpt-4o"
        elif "gpt-4-turbo" in model:
            key = "gpt-4-turbo"
        elif 'gpt-4' in model:
            key = "gpt-4"
        elif 'gpt-3.5-turbo' in model:
            key = "gpt-3.5-turbo"

        price = prompt_tokens * prices_per_million_input[
            key] + completion_tokens * prices_per_million_output[key]
        price = price / 1e6

        return price


def _encode_image_np(image_np: np.array):
    """ Encode numpy array image to bytes64 so it can be sent over http """
    assert image_np.ndim == 3 and image_np.shape[-1] == 3
    image = Image.fromarray(image_np)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())


def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return value.decode()
    return None


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def test_gpt_api():
    # model = GptAPI(do_cache=False)
    model = GptAPI()

    image = "test_images/four_cups.png"
    prompt = "whats in the image ?"
    # prompt = """\
    # This object contains which of the following objects 
    # (A) knives, (B) plates, (C) spoons, (D) cups, (E) cans, (F) towels
    # Pick one answer and return only a single letter: one of A,B,C,D,E,F.
    # """
    output = model.forward(image, prompt)



if __name__ == "__main__":
    test_gpt_api()
