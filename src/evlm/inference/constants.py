#25
DATASETS: list[str] = [
    'acevedo_et_al_2020', 'burgess_et_al_2024_contour',
    'burgess_et_al_2024_eccentricity', 'burgess_et_al_2024_texture',
    'colocalization_benchmark', "empiar_sbfsem",
    'eulenberg_et_al_2017_brightfield', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', "held_et_al_2010_galt",
    "held_et_al_2010_h2b", "held_et_al_2010_mt", 'hussain_et_al_2019',
    'icpr2020_pollen', 'jung_et_al_2022', 'kather_et_al_2016',
    'kather_et_al_2018', "kather_et_al_2018_val7k"
    'nirschl_et_al_2018', "nirschl_unpub_fluorescence", 'tang_et_al_2019',
    'wong_et_al_2022', "wu_et_al_2023"
]

CLIP_MODELS: list[str] = [
    "ALIGN", "CLIP", "BLIP", "OpenCLIP", "QuiltCLIP", "OwlVIT2", "PLIP",
    "BioMedCLIP", "ConchCLIP"
]
CHAT_MODELS: list[str] = ["CogVLM", "QwenVLM", "Kosmos2", "BLIP2", "PaliGemma"]

ALL_MODELS: list[str] = CLIP_MODELS + CHAT_MODELS
QUESTIONS = [
    'modality', 'submodality', 'domain', 'subdomain', 'stain', 'classification'
]