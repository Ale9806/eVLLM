import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures

from utils import save_output, add_alphabet, init_sub_results
import random


def evaluate_dataset(
        dataset: dict,
        model_dict: dict[str, str],
        split: str,
        transform,
        output_dir,
        prompt: str = "Answer with a single letter, no extra details.",
        question_key: str = "questions",
        do_batch_processing: bool = False,
        DEBUG: bool = True) -> None:
    """
    Evaluates a dataset using a given model.

    Parameters:
    dataset (dict): The dataset to be evaluated.
    model_dict (dict[str, str]): Dictionary containing model information.
    split (str): Split of the dataset (e.g., train, test).
    transform: Transformation function for the dataset.
    output_dir: Directory to save the output.
    prompt (str): Prompt for answering questions. Default is "Answer with a single letter, no extra details."
    DEBUG (bool): Whether to run in debug mode. Default is False.

    Returns:
    None

    We want to support parallel processing in batch if `do_batch_processing=True`
    which is valud only for GptAPI. 
    So we have a different function `process_data_point` that does the work and
    returns the result. We loop through the dataset once to generate the kwargs
    for that function, saved to `kwargs_all`
    Afterwards we actually run that function
    """
    # DEBUG = 1

    output_name = output_dir / model_dict['name'] / dataset["dataset"].name

    # store the function args and kwargs for running each question
    kwargs_all = []

    for j, data_point in enumerate(dataset["loader"]):
        questions: dict[str, str] = data_point['custom_metadata'][question_key]
        image_id: str = data_point["metadata"]['name']
        image: Path = dataset['dataset'].path / dataset[
            'dataset'].split / image_id
        sub_results: dict[str, str] = init_sub_results(data_point,
                                                       add_synonyms=True)

        for question in questions.keys():
            kwargs = dict(question=question,
                          questions=questions,
                          image=image,
                          image_id=image_id,
                          model_dict=model_dict,
                          prompt=prompt,
                          sub_results=sub_results,
                          split=split,
                          transform=transform)
            kwargs_all.append(kwargs)

        if DEBUG:
            if j == 10:
                import pdb;pdb.set_trace()
                break

    desc = f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}"

    # standard sequential processing
    if not do_batch_processing:
        results = []

        for kwargs in tqdm(kwargs_all, desc=desc):
            result = process_question(**kwargs)
            results.append(result)

    # batch processing for GPT api only
    else:
        if model_dict['name'] != "GptApi":
            raise ValueError("batch processing only valid for model 'GptApi'")

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_question, **kwargs)
                for kwargs in kwargs_all
            ]
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(kwargs_all),
                               desc=desc):
                results.append(future.result())

    save_output(results, output_name)


def process_question(question, questions, image, image_id, model_dict, prompt,
                     sub_results, split, transform):
    """A single qurestion which is one VLM call."""
    result: dict = {}
    question_str: str = questions[question]["question"]
    answer: str = questions[question]["answer"]
    options: list[str] = questions[question]["options"]
    position: int = options.index(answer)

    options, answer = add_alphabet(options, answer)
    joined_options: str = '\n'.join(options)
    template: str = prompt + "\n" + "Question:" + question_str + "\n" + joined_options

    output: dict = model_dict["model"].forward(image, template)
    result["question_class"] = question
    result["questions"] = template
    result["image_id"] = image_id
    result["correct_answer"] = answer
    result["correct_idx"] = position
    result["model_answers"] = output
    result["confidence"] = output.get('confidence', np.nan)

    for key in sub_results.keys():
        result[key] = sub_results[key]

    return result
