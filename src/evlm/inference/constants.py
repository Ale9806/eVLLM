DATASETS:list[str] = ['acevedo_et_al_2020', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', 'icpr2020_pollen',
    'nirschl_et_al_2018', 'jung_et_al_2022', 'wong_et_al_2022',
    'hussain_et_al_2019', 'colocalization_benchmark', 'kather_et_al_2016',
    'tang_et_al_2019', 'eulenberg_et_al_2017_brightfield',
    'burgess_et_al_2024_contour', 'nirschl_unpub_fluorescence',
    'burgess_et_al_2024_eccentricity', 
    'burgess_et_al_2024_texture',
    'held_et_al_2010']

DATASETS:list[str] = ['acevedo_et_al_2020', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', 'icpr2020_pollen',
    'nirschl_et_al_2018', 'jung_et_al_2022', 'wong_et_al_2022',
    'hussain_et_al_2019', 'colocalization_benchmark', 'kather_et_al_2016',
    'tang_et_al_2019', 'eulenberg_et_al_2017_brightfield',
    'burgess_et_al_2024_contour',
    'burgess_et_al_2024_eccentricity', 
    'burgess_et_al_2024_texture']

#DATASETS:list[str] = ['hussain_et_al_2019','tang_et_al_2019','icpr2020_pollen']

CLIP_MODELS:list[str] = ["ALIGN","CLIP","BLIP","OpenCLIP","QuiltCLIP","OwlVIT2","PLIP","BioMedCLIP","ConchCLIP"]
#CLIP_MODELS:list[str] = ["BLIP","PLIP","ConchCLIP"]
CHAT_MODELS:list[str] = ["CogVLM","QwenVLM","Kosmos2","BLIP2"]

ALL_MODELS: list[str] = CLIP_MODELS + CHAT_MODELS
QUESTIONS = ['modality', 'submodality', 'domain', 'subdomain' , 'stain', 'classification']