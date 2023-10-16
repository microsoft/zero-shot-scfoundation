## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os
import scanpy as sc

from typing import List, Optional, Union, Dict, Literal

import numpy as np
from scgpt.preprocess import Preprocessor

from .helpers.custom_logging import log

# switch of warnings
import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

class InputData():
    def __init__(self, 
                 adata_dataset_path: str) -> None:
        
        # check if the dataset exists
        if not os.path.isfile(adata_dataset_path):
            msg = f"Dataset {adata_dataset_path} does not exist!"
            log.error(msg)
            raise ValueError(msg)
        
        msg = f"Loading data from {adata_dataset_path}"
        log.info(msg)

        self.dataset_name = os.path.basename(adata_dataset_path).split(".")[0]
        self.adata_path = adata_dataset_path
        # read in the dataset
        self.adata = sc.read(adata_dataset_path)

        self.data_config = dict(
            data_path = adata_dataset_path,
        )
        # this will be updated if add_batch_labels is called
        self.batch_key = None
        
    def add_batch_labels(self, 
                         batch_key: Optional[str] = None,
                         batch_str_col: str = "str_batch",
                         batch_id_col: str = "batch_id") -> int:

        self.batch_key = batch_key
        self.batch_id_col = batch_id_col
        self.batch_str_col = batch_str_col

        if self.batch_key is None:
            # try guessing which column contains batch info
            # get the columns that contain "batch"
            batch_cols = [col for col in 
                            self.adata.obs.columns if "batch" in col.lower()]
            if len(batch_cols) == 1:
                ori_batch_col = batch_cols[0]
                log.info(f"Using {ori_batch_col} as batch column")
            else:
                msg = "Cannot determine which column contains batch information!"
                log.error(msg)
                raise ValueError(msg)
        else:
            ori_batch_col = self.batch_key
            log.info(f"Using {ori_batch_col} as batch column")

        self.adata.obs[self.batch_str_col] = (
            self
            .adata
            .obs[ori_batch_col]
            .astype(str)
        )
        batch_id_labels = (
            self.adata
            .obs[self.batch_str_col]
            .astype("category")
            .cat
            .codes
            .values
        )
        self.adata.obs[self.batch_id_col] = batch_id_labels
        log.debug(self.adata.obs[self.batch_id_col].value_counts())
        num_batch_types = len(set(batch_id_labels))
        log.debug(f"Number of batch types: {num_batch_types}")
        return num_batch_types

    def preprocess_data(self,
                        gene_col: str = "gene_name",
                        vocab_source: str = "model_default",
                        fract_matching: float = 0.5,
                        model_type: str = "scGPT",
                        # arguments for Geneformer preprocessing
                        gene_name_id_dict: Optional[Dict[str, str]] = None,
                        filter_gene_by_cells: Optional[int] = 10,
                        filter_cell_by_genes: Optional[int] = 10,
                        preprocessed_path: Optional[str] = None,
                        save_ext: Optional[str] = "loom",
                        # arguments for scGPT preprocessing
                        gene_vocab: Optional[List[str]] = None,
                        data_is_raw: Optional[bool] = True,
                        counts_layer: Optional[str] = "X",
                        filter_gene_by_counts: Optional[int] = 3,
                        filter_cell_by_counts: Optional[Union[int, bool]] = False,
                        n_hvg: Optional[Union[int, bool]] = 1200,
                        normalize_total: Optional[int] = 1e4,
                        n_bins: Optional[int] = 50,
                        **kwargs) -> None:

        if gene_col not in self.adata.var.columns:
            self.adata.var[gene_col] = self.adata.var.index.tolist()
            log.warning(f"Gene names not found in var columns. Using index instead.")
        
        self.gene_col = gene_col
        self.data_config["gene_col"] = gene_col
                
        # check if model_type is valid
        model_type = model_type.lower()
        valid_model_types = ["scgpt", "geneformer"]

        if model_type not in valid_model_types:
            msg = (f"Model type {model_type} not supported! "
                   f"Valid options are: {valid_model_types}.")
            log.error(msg)
            raise ValueError(msg)
        
        self.data_config["model_type"] = model_type
        self.data_config["vocab_source"] = vocab_source

        # note raw data shape
        self.data_config["input__n_cells"] = self.adata.shape[0]
        self.data_config["input__n_genes"] = self.adata.shape[1]

        # check if scgpt found in lowercase model string
        if model_type == "scgpt":

            self.data_config["data_is_raw"] = data_is_raw
            self._preprocess_data_scGPT(gene_vocab = gene_vocab,
                                        fract_matching = fract_matching,
                                        input_key = counts_layer,
                                        filter_gene_by_counts = filter_gene_by_counts,
                                        filter_cell_by_counts = filter_cell_by_counts,
                                        normalize_total = normalize_total,
                                        n_hvg = n_hvg,
                                        n_bins = n_bins,
                                        preprocessed_path = preprocessed_path,
                                        **kwargs)
            
        elif model_type == "geneformer":

            self._preprocess_data_geneformer(preprocessed_path = preprocessed_path,
                                             save_ext = save_ext,
                                             gene_name_id_dict = gene_name_id_dict,
                                             fract_matching = fract_matching,
                                             filter_cell_by_genes = filter_cell_by_genes,
                                             filter_gene_by_cells = filter_gene_by_cells)

        # note raw preprocessed shape
        self.data_config["preprocessed__n_cells"] = self.adata.shape[0]
        self.data_config["preprocessed__n_genes"] = self.adata.shape[1]

    def _preprocess_data_scGPT(self,
                               gene_vocab: List[str],
                               fract_matching: float = 0.5,
                               input_key: str = "X",
                               filter_gene_by_counts: int = 3,
                               filter_cell_by_counts: Union[int, bool] = False,
                               normalize_total: int = 1e4,
                               n_hvg: Union[int, bool] = 1200,
                               n_bins: int = 51,
                               normed_key: str = "X_normed",
                               log1p_key: str = "X_log1p",
                               binned_key: str = "X_binned",
                               preprocessed_path: Optional[str] = None) -> None:

        # preprocess the data
        self.adata.var["id_in_vocab"] = [
            1 if gene in gene_vocab else -1 
            for gene in self.adata.var[self.gene_col]
        ]
        gene_ids_in_vocab = np.array(self.adata.var["id_in_vocab"])
        fract = np.sum(gene_ids_in_vocab >= 0)/len(gene_ids_in_vocab)

        if fract < fract_matching:
            msg = f"Only {fract*100:.2f}% genes in the dataset are in the vocabulary!"
            log.error(msg)
            raise ValueError(msg)
        
        self.adata = self.adata[:, self.adata.var["id_in_vocab"] >= 0]
        self.data_config["fract_genes_in_vocab"] = fract

        log.info(
            f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)}"
            f" genes in vocabulary of size {len(gene_vocab)}."
        )
        
        if n_hvg < 1:
            n_hvg = False
        # append preprocessing parameters to run config
        d_ = {
            "preprocesing__input_key": input_key,
            "preprocesing__filter_gene_by_counts": filter_gene_by_counts,
            "preprocesing__filter_cell_by_counts": filter_cell_by_counts,
            "preprocesing__normalize_total": normalize_total,
            "preprocesing__normed_key": normed_key,
            "preprocesing__log1p_key": log1p_key,
            "preprocesing__binned_key": binned_key,
            "preprocesing__n_bins": n_bins,
            "preprocesing__n_hvg": n_hvg,
        }

        self.data_config.update(d_)

        msg = "Preprocessing data"
        log.info(msg)

        # Preprocess the data following the scGPT data pre-processing pipeline
        preprocessor = Preprocessor(
            # the key in adata.layers to use as raw data
            use_key = input_key,  
            # step 1
            filter_gene_by_counts = filter_gene_by_counts, 
            # step 2
            filter_cell_by_counts = filter_cell_by_counts, 
            # 3. whether to normalize the raw data and to what sum
            normalize_total = normalize_total,  
            # the key in adata.layers to store the normalized data
            result_normed_key = normed_key, 
            # 4. whether to log1p the normalized data
            log1p = self.data_config["data_is_raw"],  
            result_log1p_key = log1p_key,
            # 5. whether to subset the raw data to highly variable genes
            subset_hvg = n_hvg,  
            hvg_flavor = ("seurat_v3" 
                          if self.data_config["data_is_raw"] 
                          else "cell_ranger"),
            # 6. whether to bin the raw data and to what number of bins
            binning = n_bins, 
            # the key in adata.layers to store the binned data
            result_binned_key = binned_key,  
        )

        preprocessor(self.adata, batch_key = self.batch_key)
        
        if preprocessed_path is not None:
            # check if path exists
            if os.path.exists(preprocessed_path):
                msg = (f"Saving {self.dataset_name} preprocessed data "
                       f"to {preprocessed_path}")
                self.adata.write(os.path.join(preprocessed_path, 
                                              f"{self.dataset_name}.h5ad"))
            else:
                msg = (f"Directory {preprocessed_path} does not exist! "
                       "Skipping saving preprocessed data.")
                log.warning(msg)
    

    def _preprocess_data_geneformer(self,
                                    preprocessed_path: str,
                                    gene_name_id_dict: Dict[str, str],
                                    save_ext: Literal["loom", "h5ad"] = "loom",
                                    fract_matching: float = 0.5,
                                    filter_cell_by_genes: int = 10,
                                    filter_gene_by_cells: int = 10) -> None:
        
        # for geneformer we need the path to save the data, check if exists
        if preprocessed_path is None or not os.path.exists(preprocessed_path):
            msg = ("For Geneformer, preprocessed_path needs to be specified "
                   "and exists to save the dataset. Provided path: "
                   f"{preprocessed_path}")
            log.error(msg)
            raise ValueError(msg)
        
        sc.pp.calculate_qc_metrics(self.adata, 
                                   percent_top = None, 
                                   log1p = False, 
                                   inplace = True)
        self.adata.obs['n_counts'] = self.adata.obs['total_counts']
        sc.pp.filter_cells(self.adata, min_genes=int(filter_cell_by_genes))
        sc.pp.filter_genes(self.adata, min_cells=int(filter_gene_by_cells))  

        # for now, assuming gene names and using geneformer dictionary 
        # to match gene nam to ensembl id; TODO: look into better way?
        # this is tricky because ensembl ids change, in a way 
        # gene names are more constant; however they aren't necessarily unique
        # and might be missing from the geneformer dictionary/be different
        # for now, make sure to report the fraction of genes that are matched
        # and save the match/not matched
        
        self.adata.var['ensembl_id'] = self.adata.var[self.gene_col].map(gene_name_id_dict)
        self.adata.var['has_ensembl_match'] = self.adata.var['ensembl_id'].notnull()

        n_all_genes = self.adata.var.shape[0]
        n_matched = self.adata.var.has_ensembl_match.sum()
        fract = n_matched / n_all_genes

        if fract < fract_matching:
            msg = f"Only {fract*100:.2f}% genes in the dataset are in the vocabulary!"
            log.error(msg)
            raise ValueError(msg)

        # save the adata.var dataframe
        self.adata.var.to_csv(os.path.join(preprocessed_path, 
                                           f"{self.dataset_name}_var.csv"), 
                              index = False)
        
        # filter out genes that don't have a match
        self.adata = self.adata[:, self.adata.var.has_ensembl_match]

        # additionally, add the order of the samples, since they will be sorted
        # to speed up forward pass
        self.adata.obs['adata_order'] = self.adata.obs.index.tolist()

        self.data_config["fract_genes_in_vocab"] = fract

        log.info(
            f"Matched {fract*100:.2f}% genes ({n_matched}/{n_all_genes})"
            f" genes in vocabulary of size {len(gene_name_id_dict)}."
        )

        if save_ext == "loom":
            self.adata.write_loom(os.path.join(preprocessed_path, 
                                               f"{self.dataset_name}.loom"))
        elif save_ext == "h5ad":
            self.adata.write_h5ad(os.path.join(preprocessed_path, 
                                               f"{self.dataset_name}.h5ad"))

    
    def get_config(self):
        return self.data_config
    