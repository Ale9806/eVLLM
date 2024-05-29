# eVLLM (Evalauate Vision-LLMs)

A package to evaluate Vision-Lanague Models

Hello there I am a PhD student at Stanford SAIL, I am making a library to evaluate vision-language models. Why am I doing this? Well, I started benchmarking LLMs a while ago. I really benefited from having a library I could work with (vLLM) to evaluate language models, but felt a little bit frustrated when trying to do the same for vision-language models. While VLLMs are similar to LLMs, their evaluation might require some extra steps that are usually not needed for LLMs (e.g. preprocessing an out-of-distribution image with its dataset mean, instead of the pretrained mean, which is often missed by some people in the field). I am trying to abstract as much as possible to make this library useful and make sure the vision-language field benefits by having a reproducible workflow for their results.


As I am working in the intersection of CS and Biomedical applications, this package also contains biomedical VLLMs.

## Modesl Suported: 
### Base Enviorment: 

#### CLIP Style VLMs:
* ALIGN      (General)
* BLIP       (General)
* OpenCLIP   (General)
* OwlVIT     (General)
* OwlVIT2    (General)
* BiomedCLIP (Biomedical)
* PLIP       (Pathology)
* Quilt      (Pathology)

#### Generative Style VLMs:
* CogVLM         (General)
* QwenChatModel  (General)
* Kosmos2        (General)
* Fuyu           (General)
* InstructBlip   (General)


### Conch Enviorment: 
* CONCH      (Pathology)

## Installation

```bash
$ pip install evlm
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`evlm` was created by Alejandro Lozano. It is licensed under the terms of the MIT license.
`evlm` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

If you find this repo useful make sure to cite it:
```
@inproceedings{evllm,
  title={Evaluate Vision-LLMs},
  author={Alejandro Lozano},
  booktitle={Github},
  year={2024}
}
```


