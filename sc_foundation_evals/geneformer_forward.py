## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os

import importlib.util
import pickle

from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from transformers import BertForMaskedLM
from geneformer.tokenizer import TranscriptomeTokenizer

# from geneformer import EmbExtractor
from tqdm.auto import trange
from datasets import Dataset, load_from_disk
from . import utils
from .data import InputData
from .helpers.custom_logging import log

import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

def pad_tensor(t: torch.Tensor,
               max_size: int,
               pad_token_id: int = 0) -> torch.Tensor:
    """
    Pad a tensor to a max size
    """
    
    return F.pad(t, pad = (0, max_size - t.numel()), 
                 mode = 'constant', value = pad_token_id)

# get cell embeddings excluding padding
def mean_nonpadding_embs(embs, original_lens):
    # mask based on padding lengths
    mask = torch.arange(embs.size(1)).unsqueeze(0).to("cuda") < original_lens.unsqueeze(1)

    # extend mask dimensions to match the embeddings tensor
    mask = mask.unsqueeze(2).expand_as(embs)

    # use the mask to zero out the embeddings in padded areas
    masked_embs = embs * mask.float()

    # sum and divide by the lengths to get the mean of non-padding embs
    mean_embs = masked_embs.sum(1) / original_lens.view(-1, 1).float()
    return mean_embs

def average_embeddings(embs: torch.Tensor,
                       org_lengths: torch.Tensor) -> torch.Tensor:
    
    device = embs.device

    # mask based on padding lengths
    mask = (torch.arange(embs.size(1)).unsqueeze(0).to(device) < 
            org_lengths.unsqueeze(1))
    
    # extend mask dimensions to match the embeddings tensor
    if len(embs.shape) > 2:
        mask = mask.unsqueeze(2).expand_as(embs)

    # Use the mask to compute the sum over non-padded areas
    summed_embs = (embs * mask.float()).sum(dim=1)

    # Divide by the lengths to get the mean of non-padding embs
    mean_embs = summed_embs / org_lengths.view(-1, 1).float()

    return mean_embs

class Geneformer_instance():
    def __init__(self,
                 saved_model_path: Optional[str] = None,
                 model_run: str = "pretrained",
                 model_files: Dict[str, str] = {
                     "model_args": "config.json", 
                     "model_training": "training_args.bin",
                     "model_weights": "pytorch_model.bin"
                     },
                 save_dir: Optional[str] = None, 
                 explicit_save_dir: bool = False,
                 num_workers: int = 0,
                 log_wandb: bool = False,
                 project_name: str = "Geneformer_eval",
                 ) -> None:
        
        # check if the model run is supported
        supported_model_runs = ["pretrained"] #, "random", "finetune", "train"]
        if model_run not in supported_model_runs:
            msg = f"model_run must be one of {supported_model_runs}"
            log.error(msg)
            raise ValueError(msg)
        self.model_run = model_run

        self.saved_model_path = saved_model_path
        self.model_files = model_files

        if num_workers == -1:
            num_workers = len(os.sched_getaffinity(0))

        if num_workers == 0:
            num_workers = 1

        self.num_workers = num_workers

        # check if output directory exists
        if save_dir is not None:
            if explicit_save_dir:
                self.output_dir = save_dir
            else:
                self.output_dir = os.path.join(save_dir,
                                               self.run_id)
                # if the top out directory does not exist, create it
                if not os.path.exists(save_dir):
                    log.warning(f"Creating the top output directory {save_dir}")
                    os.makedirs(save_dir)
        else:
            # save in a current path
            self.output_dir = os.path.join(os.getcwd(), self.run_id)

        # if the out directory already exists, raise an error
        if os.path.exists(self.output_dir) and not explicit_save_dir:
            msg = f"Output directory: {self.output_dir} exists. Something is wrong!"
            log.error(msg)
            raise ValueError(msg)
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" 
                                   if torch.cuda.is_available()
                                   else "cpu")
        
        log.info(f"Using device {self.device}")

        self.project_name = project_name
        if log_wandb:
            has_wandb = importlib.util.find_spec("wandb") is not None
            if not has_wandb:
                msg = "Wandb is not installed. Please install wandb to log to wandb."
                log.error(msg)
                raise RuntimeError(msg)
            if has_wandb:
                import wandb
            self._wandb = wandb
        else: 
            self._wandb = None

        # update this when saved config so that when training it only is saved once
        self.config_saved = False

    def _check_attr(self, 
                    attr: str, 
                    not_none: bool = True) -> bool:
        """
        Check if the argument is in the class
        """
        out = hasattr(self, attr)
        if not_none and out:
            out = getattr(self, attr) is not None
        return out

    def load_pretrained_model(self) -> None:
        
        self.model = BertForMaskedLM.from_pretrained(self.saved_model_path,
                                                     output_attentions=False,
                                                     output_hidden_states=True)
        self.model = self.model.to(self.device)
        log.info(f"Model successfully loaded from {self.saved_model_path}")
    
    
    def load_tokenized_dataset(self,
                               dataset_path: str) -> None:
        
        self.tokenized_dataset = load_from_disk(dataset_path)

    def tokenize_data(self,
                      adata_path: str,
                      dataset_path: str,
                      cell_type_col: str = "cell_type",
                      columns_to_keep: List[str] = ["adata_order"]):
        
        dataset_name = os.path.basename(adata_path).split(".")[0]
        
        cols_to_keep = dict(zip([cell_type_col] + columns_to_keep, 
                                ['cell_type'] + columns_to_keep))
        # initialize tokenizer
        self.tokenizer = TranscriptomeTokenizer(cols_to_keep, 
                                                nproc = self.num_workers)

        # get the extension from adata_path
        _, ext = os.path.splitext(adata_path)
        ext = ext.strip(".")

        if ext not in ["loom", "h5ad"]:
            msg = f"adata_path must be a loom or h5ad file. Got {ext}"
            log.error(msg)
            raise ValueError(msg)
        
        if ext == "h5ad":
            msg = ("using h5ad file. This sometimes causes issues. "
                   "If not working try with loom.")
            log.warning(msg)
        
        # get the top directory of the adata_path
        adata_dir = os.path.dirname(adata_path)

        self.tokenizer.tokenize_data(adata_dir,
                                     dataset_path, 
                                     dataset_name,
                                     file_format=ext)

        
        # tokenizer does not return the dataset
        # load the dataset
        self.load_tokenized_dataset(os.path.join(dataset_path, 
                                                 f"{dataset_name}.dataset"))


    def load_vocab(self,
                   dict_paths: str) -> None:
        
        token_dictionary_path = os.path.join(dict_paths,
                                            "token_dictionary.pkl")
        with open(token_dictionary_path, "rb") as f:
            self.vocab = pickle.load(f)
        
        self.pad_token_id = self.vocab.get("<pad>")
        
        # size of vocabulary
        self.vocab_size = len(self.vocab)  

        gene_name_id_path = os.path.join(dict_paths,
                                            "gene_name_id_dict.pkl")
        with open(gene_name_id_path, "rb") as f:
            self.gene_name_id = pickle.load(f)
    

    def _extend_batch(self,
                      batch_dataset: Dataset,
                      return_attention_mask: bool = True):
        
        max_size = max(batch_dataset['length'])
        
        batch_ = [pad_tensor(x, max_size, self.pad_token_id) 
                  for x in batch_dataset['input_ids']]
        
        batch_ = torch.stack(batch_).to(self.device)

        if return_attention_mask:
            mask_ = [[1] * l + [0] * (max_size - l) 
                     for l in batch_dataset['length']]
            mask_ = torch.tensor(mask_).to(self.device)
            return batch_, mask_
            
        return batch_
    
    def _pass_batch(self, 
                    batch_ids: torch.Tensor, 
                    attention_mask: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        # make sure that batch and attn_mask on the same device
        batch_ids = batch_ids.to(self.device)
        attn_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids = batch_ids,
                                 attention_mask = attn_mask,
                                 **kwargs)
        
        return outputs
    

    def extract_embeddings(self,
                           data: InputData,
                           batch_size: int = 48,
                           embedding_key: str = "geneformer",
                           layer: int = -2):

        # check if tokenized dataset is loaded
        if not self._check_attr("tokenized_dataset"):
            msg = "Tokenized dataset not loaded. Please load the tokenized dataset."
            log.error(msg)
            raise RuntimeError(msg)
        
        # check if layer is valid
        n_layers = self.model.config.num_hidden_layers
        if layer >= n_layers or layer < -n_layers:
            msg = (f"Layer {layer} is not valid. There are only {n_layers} "
                   f"Acceptable values are between {-n_layers} (if counting "
                   f"forwards) and {n_layers - 1} (if counting backwards)")
            log.error(msg)
            raise ValueError(msg)

        # save the embeddings to subdir
        embeddings_subdir = os.path.join(self.output_dir, "model_outputs")
        os.makedirs(embeddings_subdir, exist_ok=True)

        cell_embs_list = []
        rankings_list = []

        size = len(self.tokenized_dataset)

        for i in trange(0, size, batch_size, 
                        desc = "Geneformer (extracting embeddings)"):
            
            max_range = min(i+batch_size, size)
            batch_dataset = self.tokenized_dataset.select(list(range(i, max_range)))
            batch_dataset.set_format(type = 'torch')

            org_lengths = torch.tensor(batch_dataset['length']).to(self.device)
            
            batch, attn_mask = self._extend_batch(batch_dataset)

            model_output = self._pass_batch(batch, 
                                            attention_mask = attn_mask)
            
            embs = model_output.hidden_states[layer]

            # cell_embs = average_embeddings(embs, org_lengths)
            cell_embs = mean_nonpadding_embs(embs, org_lengths)
            
            # add cell embeddings to the list
            cell_embs_list.extend(cell_embs.detach().cpu().numpy())

            # now, get the ranking reconstruction
            out_rankings = (model_output.logits
                            .argmax(axis=-1)
                            .detach().cpu().numpy())
            
            # save the rankings with the original order
            rankings_list.extend(out_rankings)
            
            torch.cuda.empty_cache()
            del model_output
            del batch
            del attn_mask
            del embs
            del cell_embs
            
        self.cell_embeddings = np.array(cell_embs_list)

        self.output_rankings = rankings_list
        self.input_rankings = [np.array(item) 
                               for item 
                               in self.tokenized_dataset['input_ids']]

        # add embeddings to adata
        data.adata.obsm[embedding_key] = self.cell_embeddings

        # for plotting later, save the data.adata.obs
        # order here agrees with the order of the embeddings
        data.adata.obs.to_csv(os.path.join(embeddings_subdir, 
                                           "adata_obs.csv"))



