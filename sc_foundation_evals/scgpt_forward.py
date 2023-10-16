## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os
import json
import time
import importlib.util

from typing import Dict, Optional, List, Union

from scipy.sparse import issparse
import torch
import scanpy as sc
import numpy as np
from torch.utils.data import Dataset, DataLoader

# import utils from scgpt_eval
from . import utils
from .data import InputData

from scgpt import SubsetsBatchSampler
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from scgpt.utils import set_seed

from .helpers.custom_logging import log

# create a helper function to sanitize the name
def sanitize_name(old_name):
    """Sanitize the name of the label 
    to be used as variable name in annotation"""
    import re
    # remove leading and trailing whitespace
    old_name = old_name.strip()
    # remove non-alphanumeric characters
    old_name = re.sub(r"[^a-zA-Z0-9\s_-]", "", old_name)
    # split on " ", "_", and "-"
    words = re.split(r"[\s_-]", old_name)
    # return camel case
    # return words[0] + ''.join(word.capitalize() for word in words[1:])

    # return lowercase words joined by _
    return '_'.join( word.lower() for word in words)

class SeqDataset(Dataset):
    def __init__(self, 
                 data: Dict[str, torch.Tensor],
                 gene_key: str = "gene_ids"):
        self.data = data
        self.gene_key = gene_key

    def __len__(self):
        return self.data[self.gene_key].shape[0]

    def __getitem__(self, idx):
        d_ = {k: v[idx] for k, v in self.data.items()}
        d_['idx'] = idx
        return d_
    

class scGPT_instance():
    def __init__(self,
                 saved_model_path: Optional[str] = None,
                 model_run: str = "pretrained",
                 model_files: Dict[str, str] = {
                    "model_args": "args.json", 
                    "model_vocab": "vocab.json",
                    "model_weights": "best_model.pt"
                 },
                 batch_size: int = 8,
                 save_dir: Optional[str] = None, 
                 explicit_save_dir: bool = False,
                 num_workers: int = 0,
                 n_log_reports: int = 10,
                 log_wandb: bool = False,
                 project_name: str = "scGPT_eval",
                 ) -> None:
        
        # check if the model run is supported
        # add "train" "finetune" later
        supported_model_runs = ["pretrained", "random"] 
        if model_run not in supported_model_runs:
            msg = f"model_run must be one of {supported_model_runs}"
            log.error(msg)
            raise ValueError(msg)
        self.model_run = model_run
        
        if self.model_run in ["pretrained", "finetune"] and saved_model_path is None:
            msg = "saved_model_path must be provided if model_run is not 'train'"
            log.error(msg)
            raise ValueError(msg)
        
        if self.model_run == "train" and saved_model_path is not None:
            msg = "args from saved_model_path will be used for training the model"
            log.warning(msg)
    
        self.saved_model_path = saved_model_path

        self.model_files = model_files
        if batch_size % 8 != 0:
            batch_size_ = batch_size
            batch_size = (batch_size // 8 + 1) * 8
    
            msg = ("Using AMP by default (currently hardcoded) "
                   f"batch_size must be a multiple of 8 "
                   f"provided {batch_size_}, changing to {batch_size}")
            log.warning(msg)
        
        self.batch_size = batch_size
        

        self.run_id = (f'{self.model_run}__'
                       f'{time.strftime("%Y-%m-%d_%H-%M-%S")}')


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

        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")


        self.num_workers = num_workers
        self.n_log_reports = n_log_reports

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

    def create_configs(self, 
                       seed: int = 97,
                       # ---> scGPT_human defaults
                       embsize: int = 512,
                       nheads: int = 8,
                       d_hid: int = 512,
                       nlayers: int = 12,
                       nlayers_cls: int = 3,
                       dropout: float = 0.2,
                       pad_token: str = "<pad>",
                       pad_value: int = -2,
                       mask_value: int = -1,
                       mask_ratio: Union[float, List[float]] =[0.25, 0.5, 0.75],
                       do_mvc: bool = True, # MVC in args.json
                       input_emb_style: str = "continuous",
                       n_bins: Optional[int] = 51,
                       use_fast_transformer: bool = True,
                       # <--- scGPT_human defaults
                       # ---> scgpt.TransformerModel class default
                       n_cls: int = 1, 
                       do_dab: bool = False, 
                       use_batch_labels: bool = False, 
                       domain_spec_batchnorm: bool = False, 
                       cell_emb_style: str = "cls", 
                       mvc_decoder_style: str = "inner product", 
                       ecs_threshold: float = 0.3, 
                       explicit_zero_prob: bool = False, 
                       fast_transformer_backend: str = "flash", 
                       pre_norm: bool = False, 
                       # <--- scgpt.TransformerModel class default
                       do_cce: bool = False,
                       do_ecs: bool = False,
                       do_cls: bool = False,
                       max_seq_len: int = 1200,
                       per_seq_batch_sample: bool = False, 
                       shuffle: bool = False,
                       append_cls: bool = True,
                       permute_gene_order: bool = True):
        # input_emb_style ["category", "continuous", "scaling"] 
        # check if input_emb_style is supported
        supported_input_emb_styles = ["category", "continuous", "scaling", "concat"]
        # "category" => CategoryValueEncoder 
        # coded here: https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/model/model.py#L778
        # "continous" => ContinuousValueEncoder 
        # coded here: https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/model/model.py#L748
        # "concat" added by me for concatenating the gene token embeddings 
        # and the expression value embedding
        # nn.Identity()
        if input_emb_style not in supported_input_emb_styles:
            msg = f"input_emb_style must be one of {supported_input_emb_styles}"
            log.error(msg)
            raise ValueError(msg)

        if append_cls:
            self.cell_emb_position = 0
        # check if cell_emb_style is supported
        supported_cell_emb_styles = ["cls", "avg-pool", "w-pool"]
        if cell_emb_style not in supported_cell_emb_styles:
            msg = f"cell_emb_style must be one of {supported_cell_emb_styles}"
            log.error(msg)
            raise ValueError(msg)
    
        # model config
        self.model_config = dict(
            # ---> arguments used in TransformerModel init
            embsize = embsize, # controls d_hid arg in model
            nheads = nheads, # nheads arg in model
            d_hid = d_hid, # d_hid arg in model
            nlayers = nlayers, 
            nlayers_cls = nlayers_cls, 
            n_cls = n_cls,
            dropout = dropout,
            pad_token = pad_token, 
            pad_value = pad_value,
            mask_value = mask_value,
            mask_ratio = mask_ratio,
            do_mvc = do_mvc, # also known as GEPC
            do_cls = do_cls,
            do_cce = do_cce,
            do_ecs = do_ecs,
            do_dab = do_dab,
            use_batch_labels = use_batch_labels,
            # update this with # of batch labels from data config 
            num_batch_labels = None,  
            domain_spec_batchnorm = domain_spec_batchnorm, # False
            input_emb_style = input_emb_style, 
            n_bins = n_bins, # n_input_bins in model
            cell_emb_style = cell_emb_style, 
            mvc_decoder_style = mvc_decoder_style,
            ecs_threshold = ecs_threshold, 
            explicit_zero_prob = explicit_zero_prob, 
            use_fast_transformer = use_fast_transformer,
            fast_transformer_backend = fast_transformer_backend,
            pre_norm = pre_norm,
            # <--- arguments used in TransformerModel init
            # Flag to indicate whether to update model parameters during training
            do_train = (True 
                        if self.model_run in ["finetuned", "train", "retrain"] 
                        else False), 
            # Default setting: Automatic Mixed Precision,
            amp = True,  
            # what model are we using
            model_run = self.model_run,
            # Path to pre-trained model configs and weights
            load_model = self.saved_model_path, 
            # special tokens
            special_tokens = [pad_token, "<cls>", "<eoc>"],
        )
        
        # TODO: work on the logical split of configs
        self.run_config = dict(
            save_dir = self.output_dir,
            per_seq_batch_sample = per_seq_batch_sample,
            shuffle = shuffle,
            # max sequence length
            max_seq_len = max_seq_len,
            append_cls = append_cls,
            seed = seed,
            permute_gene_order = permute_gene_order,
        )

        if self._wandb:
            if self._wandb.run is None:
                self._wandb.init(
                    project = os.getenv("WANDB_PROJECT", self.project_name), 
                    name = os.getenv("WANDB_RUN_NAME", self.run_id),
                    config = {**self.model_config, **self.run_config},
                    dir = self.output_dir
                    )



    def update_config(self) -> None:
        if self.saved_model_path is None and self.model_run != "train":
            msg = "saved_model_path must be provided if model_run is not 'train'"
            log.error(msg)
            raise ValueError(msg)
        
        if self.saved_model_path:
            with open(os.path.join(self.saved_model_path, 
                                self.model_files['model_args']), "r") as f:
                pre_trained_config = json.load(f)
            
            # update the config with the updated config
            for key, value in pre_trained_config.items():
                if key in self.model_config.keys():
                    if value != self.model_config[key] and key not in ['model_run', 'load_model']:
                        # print a warning if the value is different
                        log.warning(f"Overriding model config['{key}']"
                                    f" with {value}"
                                    f" (was {self.model_config[key]})")
                        self.model_config[key] = value
                else:
                    if key in self.run_config:
                        log.warning(f"Overriding pre-trained config['{key}']"
                                    f" with {self.run_config[key]}"
                                    f" (was {value})")
                        self.model_config[key] = self.run_config[key]
        else:
            msg = "saved_model_path is not provided. Nothing to update."
            log.warning(msg)

        # remove dist_url or save_dir from model_config
        # those are artifacts of scGPT config
        for key in ["dist_url", "save_dir"]:
            if key in self.model_config.keys():
                del self.model_config[key]
        
        
    def load_vocab(self, 
                   vocab_file: str = None) -> None:
        
        if vocab_file is None:
            vocab_file = os.path.join(self.saved_model_path, 
                                      self.model_files['model_vocab'])
        
        self.model_config['vocab_path'] = vocab_file
        # check if file exists
        if not os.path.exists(vocab_file):
            msg = f"Vocab file {vocab_file} does not exist!"
            log.error(msg)
            raise FileNotFoundError(msg)
        
        msg = f"Loading vocab from {vocab_file}"
        log.info(msg)

        # load the vocab
        self.vocab = GeneVocab.from_file(vocab_file)
        for s in self.model_config['special_tokens']: # type: ignore
            if s not in self.vocab:
                self.vocab.append_token(s)

        self.vocab.set_default_index(self.vocab[self.model_config['pad_token']]) # type: ignore
        self.model_config['ntokens'] = len(self.vocab)  # size of vocabulary

    def initialize_model(self) -> None:
        # set seed
        set_seed(self.run_config['seed'])
        
        # check if vocab is loaded
        if self.vocab is None:
            msg = "Vocab not loaded!"
            log.error(msg)
            raise ValueError(msg)

        if self.model_config["use_batch_labels"] and self.model_config["num_batch_labels"] is None:
            msg = ("Model configured to use batch labels but number of batches "
                   "not set! If use_batch_labels=True, load data before "
                   "initializing model to count the number of batches.")
            
            log.error(msg)
            raise ValueError(msg)   

        log.debug(f"Use fast transformer? {self.model_config['use_fast_transformer']}")
        # annoyingly this triggers VSCode to show an error
        self.model = TransformerModel(
            ntoken = self.model_config['ntokens'],
            d_model = self.model_config['embsize'],
            nhead = self.model_config['nheads'],
            d_hid = self.model_config['d_hid'],
            nlayers = self.model_config['nlayers'],
            nlayers_cls = self.model_config['nlayers_cls'], 
            n_cls = self.model_config['n_cls'],
            vocab = self.vocab, 
            dropout = self.model_config['dropout'], 
            pad_token = self.model_config['pad_token'], 
            pad_value = self.model_config['pad_value'], 
            do_mvc = self.model_config['do_mvc'], 
            do_dab = self.model_config['do_dab'], 
            use_batch_labels = self.model_config['use_batch_labels'],
            num_batch_labels = self.model_config['num_batch_labels'],
            domain_spec_batchnorm = self.model_config['domain_spec_batchnorm'], 
            input_emb_style = self.model_config['input_emb_style'], 
            n_input_bins = self.model_config['n_bins'], 
            cell_emb_style = self.model_config['cell_emb_style'], 
            mvc_decoder_style = self.model_config['mvc_decoder_style'], 
            ecs_threshold = self.model_config["ecs_threshold"],
            explicit_zero_prob = self.model_config["explicit_zero_prob"], 
            use_fast_transformer = self.model_config["use_fast_transformer"], 
            fast_transformer_backend = self.model_config["fast_transformer_backend"],
            pre_norm = self.model_config['pre_norm']
        )
        
        # if wandb set, log the model
        if self._wandb:
            self._wandb.watch(self.model)
    
    
    def load_pretrained_model(self) -> None:
        # check if configs are created
        if self.model_config is None:
            msg = "Model config not created!"
            log.error(msg)
            raise ValueError(msg)

        self.update_config()
        self.load_vocab()
        self.initialize_model()

        model_file = os.path.join(self.saved_model_path, 
                                  self.model_files['model_weights'])
        
        msg = f"Loading model from {model_file}"
        log.info(msg)
        try:
            self.model.load_state_dict(torch.load(model_file))
            log.debug(f"Loading all model params from {model_file}")
        except:
            log.warning(f"Loading partial model params from {model_file}")
            # only load params that are in the model and match the size
            model_dict = self.model.state_dict()
            pretrained_dict_full = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict_full.items()
                if k in model_dict and v.shape  ==  model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                log.debug(f"Loading params {k} with shape {v.shape}")
            
            # print which params are not loaded
            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    log.warning(f"Cannot load {k} with shape {v.shape}")

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            if torch.cuda.device_count() > 1:
                log.info(f"Using {torch.cuda.device_count()} GPUs")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)

    def randomly_initialize_model(self) -> None:
        self.update_config()
        self.load_vocab()
        self.initialize_model()

        msg = f"Randomly initializing model"
        log.info(msg)

        if torch.cuda.device_count() > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)


    # data_loader
    @staticmethod
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        per_seq_batch_sample: bool = False,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Prepare a dataloader from a data_pt

        Args:
            data_pt:                A dictionary with elements such as tokenized 
                                    gene_ids, values, etc.
            batch_size:             Batch size
            per_seq_batch_sample:   If True, sample from each batch of sequences
                                    instead of each sequence
            shuffle:                If True, shuffle the data
            intra_domain_shuffle:   If True, shuffle the data within each batch
            drop_last:              If True, drop the last batch if it is 
                                    smaller than batch_size
            num_workers:            Number of workers for the dataloader; 
                                    if -1, use the number of available CPUs; 
                                    positive integers turn multiprocessing on
        Returns:
            A DataLoader object
        """
        
        if num_workers == -1:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = (
                    np.where(batch_labels_array == batch_label)[0]
                    .tolist()
                )
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset = dataset,
                batch_sampler = SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle = intra_domain_shuffle,
                    inter_subset_shuffle = shuffle,
                    drop_last = drop_last,
                ),
                num_workers = num_workers,
                pin_memory = True,
            )
            return data_loader
    
        data_loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers,
            pin_memory = True,
        )
        return data_loader
    
    def tokenize_data(self, 
                      data: InputData,
                      input_layer_key: str = "X_binned",
                      include_zero_genes: bool = False) -> None:
        
        
        # check if data is preprocessed
        if input_layer_key not in data.adata.layers.keys():
            msg = f"{input_layer_key} is not in adata.layers! Preprocess the data"
            log.error(msg)
            raise ValueError(msg)

        input_data = (
                data.adata.layers[input_layer_key].A
                if issparse(data.adata.layers[input_layer_key])
                else data.adata.layers[input_layer_key]
            )
        
        self.genes = data.adata.var[data.data_config["gene_col"]].values.tolist()
        self.gene_ids = np.array(self.vocab(self.genes), dtype=int)

        self.data_config = data.get_config()
        # add the arguments of tokenizer to config
        self.run_config['tokenizer__input_layer_key'] = input_layer_key
        self.run_config['tokenizer__include_zero_genes'] = include_zero_genes

        msg = "Tokenizing data"
        log.info(msg)
        
        self.tokenized_data = tokenize_and_pad_batch(
            input_data,
            self.gene_ids,
            max_len = self.run_config['max_seq_len'],
            vocab = self.vocab,
            pad_token = self.model_config['pad_token'],
            pad_value = self.model_config['pad_value'],
            # append <cls> token at the beginning
            append_cls = self.run_config['append_cls'],  
            include_zero_gene = include_zero_genes,
        )

        if data.batch_key is not None:
            batch_labels = (
                data
                .adata
                .obs[data.batch_id_col]
                .values
                )
            batch_labels = torch.from_numpy(np.array(batch_labels)).long()
            self.tokenized_data["batch_labels"] = batch_labels
        
        self.tokenized_data["values"] = self.tokenized_data["values"].float()


    def get_dataloader(self, 
                       input_layer_key: str = "X_binned",
                       include_zero_gene: bool = False, 
                       drop_last: bool = False) -> None:

        # check if data is tokenized and make sure include zero genes is shared
        if not self._check_attr("tokenized_data"):
            self.tokenize_data(input_layer_key = input_layer_key,
                               include_zero_gene = include_zero_gene)
        else:
            include_zero_gene = self.run_config['tokenizer__include_zero_genes']

        data_pt = {
            "gene_ids": self.tokenized_data["genes"],
            "values": self.tokenized_data["values"]
        }

        if self.model_config["use_batch_labels"]:
            data_pt["batch_labels"] = self.tokenized_data["batch_labels"]
        

        msg = "Preparing dataloader"
        log.info(msg)

        self.data_loader = self.prepare_dataloader(
            data_pt,
            batch_size = self.batch_size,
            shuffle = self.run_config['shuffle'],
            drop_last = drop_last,
            num_workers = self.num_workers,
            per_seq_batch_sample = self.run_config['per_seq_batch_sample']
        )

    def save_config(self):
        msg = f"Saving config to {self.output_dir}"
        log.info(msg)

        with open(os.path.join(self.output_dir, "args.json"), "w") as f:
            json.dump(self.model_config, f, indent=4)

        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.run_config, f, indent=4)

        # check if data config is created, if is, save it
        if self._check_attr("data_config"):
            with open(os.path.join(self.output_dir, "data_config.json"), "w") as f:
                json.dump(self.data_config, f, indent=4)
        else:
            msg = "Data config not created."
            log.warning(msg)

        self.config_saved = True

    def extract_embeddings(self,
                           data: InputData,
                           embedding_key: str = "X_scGPT",
                           experimental: bool = False,
                           ) -> Optional[Dict[str, np.ndarray]]:
        
        # check if model is loaded 
        if not self._check_attr("model"):
            msg = "Please load model before extracting embeddings!"
            log.error(msg)
            raise ValueError(msg)
        
        # check if data loader is created
        if not self._check_attr("data_loader"):
            self.get_dataloader()
        
        self.model.eval()
        if not self.config_saved:
            self.save_config()

        # save the embeddings to subdir
        embeddings_subdir = os.path.join(self.output_dir, "model_outputs")
        os.makedirs(embeddings_subdir, exist_ok=True)

        # update wandb config
        if self._wandb:
            self._wandb.config.update(self.model_config,
                                      # some config is updated after init
                                      allow_val_change = True)
            self._wandb.config.update(self.run_config,
                                      # some config is updated after init
                                      allow_val_change=True)
        
        msg = "Extracting embeddings"
        log.info(msg)

        cell_embeddings = []
        mlm_output = []
        batch_idxs = []
        mvc_output = []
        masked_values = []

        # how many updates to log
        login_freq = len(self.data_loader) // self.n_log_reports

        for batch, batch_data in enumerate(self.data_loader):
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            if experimental:
                # mask 20% of the values with self.model_config['mask_value']
                target_values = input_values.clone()
                # for each row in the input values, mask 20% of the values
                # Apply masking row-wise
                for i in range(input_values.shape[0]):
                    row = input_values[i, :]
                    positive_indices = torch.nonzero(row > 0)

                    # calculate how many values we need to change
                    n_to_change = int(0.2 * len(positive_indices))

                    # if the row has no positive values or the number of positive values is less than 5, 
                    # it's impossible to replace 20% of them
                    if n_to_change == 0:
                        continue

                    # choose random indices to change
                    indices_to_change = torch.randperm(len(positive_indices))[:n_to_change]

                    # replace chosen values with -1
                    input_values[i, 
                                 positive_indices[indices_to_change]] = self.model_config['mask_value']
            
            if self.model_config["use_batch_labels"]:
                # TODO: I'm not using this, should delete it and add only if needed
                batch_labels = batch_data["batch_labels"].to(self.device)
            
            # for when used with shuffling
            batch_idx = batch_data["idx"].numpy()
            batch_idxs.append(batch_idx)

            # permute the gene order for each sample
            if self.run_config["permute_gene_order"]:
                input_values_temp = input_values.clone()
                input_values, indx = utils.permute_values(input_values)
                input_gene_ids = utils.rearrange(input_gene_ids, indx)
                if experimental:
                    target_values = utils.rearrange(target_values, indx)

            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.model_config['pad_token']])
            
            # TODO: added this because there is a bug in the code
            self.model = self.model.to(self.device)

            if batch % login_freq == 0:
                msg = f"Extracting embeddings for batch {batch+1}/{len(self.data_loader)}"
                log.info(msg)

            with torch.no_grad() and torch.cuda.amp.autocast(enabled=self.model_config['amp']):
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask = src_key_padding_mask,
                    batch_labels = batch_labels if self.model_config["use_batch_labels"] else None,
                    # gene expression prediction from cell embedding? GEPC
                    MVC = self.model_config["do_mvc"],
                    # elastic cell similarity
                    ECS = self.model_config["do_ecs"], 
                    # cell type classification objective
                    CLS = self.model_config["do_cls"], 
                    # contrastive cell embedding objective
                    CCE = self.model_config["do_cce"]
                )

                # revert order, so that it works with down stream tasks
                if self.run_config["permute_gene_order"]:
                    # inputs
                    input_values = utils.reverse_permute(input_values, indx)
                    # is the reverse permuted exactly the same as the original?
                    assert torch.all(input_values == input_values_temp)
                    input_gene_ids = utils.reverse_permute(input_gene_ids, indx)
                    if experimental:
                        target_values = utils.reverse_permute(target_values, indx)

                    output_dict["mlm_output"] = (
                        utils.reverse_permute(output_dict["mlm_output"], 
                                              indx)
                    )

                    if self.model_config["do_mvc"]:
                        output_dict["mvc_output"] = (
                            utils.reverse_permute(output_dict["mvc_output"], 
                                                  indx)
                        )

                cell_embeddings.append(output_dict["cell_emb"]
                                       .detach()
                                       .cpu()
                                       .numpy())
                mlm_output.append(output_dict["mlm_output"]
                                  .detach()
                                  .cpu()
                                  .numpy())
                if self.model_config["do_mvc"]:
                    mvc_output.append(output_dict["mvc_output"]
                                      .detach()
                                      .cpu()
                                      .numpy())
                if experimental:
                    masked_values.append(input_values.eq(self.model_config['mask_value'])
                                         .detach()
                                         .cpu()
                                         .numpy())
        # flatten the list of cell embeddings
        self.cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        # normalize cell embeddings
        self.cell_embeddings = self.cell_embeddings / np.linalg.norm(
            self.cell_embeddings, axis = 1, keepdims = True
        )
        # flatten the list of mlm_output
        self.mlm_output = np.concatenate(mlm_output, axis=0)

        if self.model_config["do_mvc"]:
            self.mvc_output = np.concatenate(mvc_output, axis=0)
        else:
            self.mvc_output = None

        if experimental:
            self.masked_values = np.concatenate(masked_values, axis=0)

        self.batch_indices = np.concatenate(batch_idxs, axis=0)
        
        # revert to original order if shuffling is used
        sorted = all(self.batch_indices[i] <= self.batch_indices[i+1] 
                     for i in range(len(self.batch_indices)-1))

        if not sorted:
            self.cell_embeddings = self.cell_embeddings[np.argsort(self.batch_indices)]
            self.mlm_output = self.mlm_output[np.argsort(self.batch_indices)]
            if self.model_config["do_mvc"]:
                self.mvc_output = self.mvc_output[np.argsort(self.batch_indices)]
            if experimental:
                self.masked_values = self.masked_values[np.argsort(self.batch_indices)]

        # save the gene ids to a gzipped numpy file
        np.savez_compressed(
            os.path.join(embeddings_subdir, "out.npz"),
            gene_ids =  self.tokenized_data["genes"].detach().cpu().numpy(),
            masked_values = self.masked_values if experimental else None,
            values = self.tokenized_data["values"].detach().cpu().numpy(),
            mlm_output = self.mlm_output,
            mvc_output = self.mvc_output,
            cell_embeddings = self.cell_embeddings
        )

        # add embeddings to adata
        data.adata.obsm[embedding_key] = self.cell_embeddings

        # for plotting later, save the data.adata.obs
        # order here agrees with the order of the embeddings
        data.adata.obs.to_csv(os.path.join(embeddings_subdir, 
                                           "adata_obs.csv"))


    def clean_up(self, 
                 save_model: bool = False) -> None:
        # close wandb
        if self._wandb:
            self._wandb.finish()
        
        if save_model:
            import pickle
            # save the model
            with open(os.path.join(self.output_dir, "model.pkl"), "wb") as f:
                pickle.dump(self.model, f)
