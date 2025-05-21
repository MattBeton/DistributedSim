import os, subprocess, json, argparse
import requests

import torch

from .ablang_2 import ablang as ablang_2_model
from .ablang_2 import tokenizers as ablang_2_tokenizer

available_ablang_1_models = ["ablang-1-heavy", "ablang-1-light"]
available_ablang_2_models = ["ablang-paired"]



def load_model(model_to_use, random_init=False, device='cpu'):
    
    if model_to_use in available_ablang_1_models:
        
        chain = "heavy" if "heavy" in model_to_use else "light"
        AbLang, tokenizer, hparams = fetch_ablang_1_model(chain, random_init=random_init, device=device)
    else:
        AbLang, tokenizer, hparams = fetch_ablang_2_model(model_to_use, random_init=random_init, device=device)

    return AbLang, tokenizer, hparams
    
        
# def fetch_ablang_1_model(chain, random_init=False, device='cpu'):
    
#     from .ablang_1 import model as ablang_1_model
#     from .ablang_1 import tokenizers as ablang_1_tokenizer
    
#     # Download model to specific place - if already downloaded use it without downloading again
#     model_folder = os.path.join(os.path.dirname(__file__), "model-weights-{}".format(chain))
#     os.makedirs(model_folder, exist_ok = True)

#     if not os.path.isfile(os.path.join(model_folder, "amodel.pt")):
#         print("Downloading model ...")

#         url = "http://opig.stats.ox.ac.uk/website/data/downloads/ablang-{}.tar.gz".format(chain)
#         tmp_file = os.path.join(model_folder, "tmp.tar.gz")

#         with open(tmp_file,'wb') as f: f.write(requests.get(url).content)

#         subprocess.run(["tar", "-zxvf", tmp_file, "-C", model_folder], check = True) 

#         os.remove(tmp_file)
        
#     with open(os.path.join(model_folder, 'hparams.json'), 'r', encoding='utf-8') as f:
#         hparams = argparse.Namespace(**json.load(f))    

#     AbLang = ablang_1_model.AbLang(hparams)

#     if not random_init:
#         AbLang.load_state_dict(torch.load(os.path.join(model_folder, 'amodel.pt'), map_location=torch.device(device)))
        
#     tokenizer = ablang_1_tokenizer.ABtokenizer(os.path.join(model_folder, 'vocab.json'))
        
#     return AbLang, tokenizer, hparams


def fetch_ablang_2_model(model_to_use, random_init=False, device='cpu'):
    
    with open(os.path.join(model_to_use, 'hparams.json'), 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**json.load(f))    

    if not 'use_moe' in hparams:
        hparams.use_moe = False
        
        
    AbLang = ablang_2_model.AbLang(
        vocab_size = hparams.vocab_size,
        hidden_embed_size = hparams.hidden_embed_size,
        n_attn_heads = hparams.n_attn_heads,
        n_encoder_blocks = hparams.n_encoder_blocks,
        padding_tkn = hparams.pad_tkn,
        mask_tkn = hparams.mask_tkn,
        layer_norm_eps = hparams.layer_norm_eps,
        dropout = hparams.dropout, 
        use_tkn_dropout = hparams.use_tkn_dropout,
        a_fn = hparams.a_fn,
        use_moe = hparams.use_moe,
    )

    if not random_init:
        AbLang.load_state_dict(torch.load(os.path.join(model_to_use, 'model.pt'), map_location=torch.device(device)))
    
    tokenizer = ablang_2_tokenizer.ABtokenizer()
    
    return AbLang, tokenizer, hparams
    
