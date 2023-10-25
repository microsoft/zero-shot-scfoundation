## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

from typing import Dict, Union, List
from anndata import AnnData
import numpy as np
import scanpy as sc
import scib
import torch
import torch.nn.functional as F

from .helpers.custom_logging import log

# MODIFIED wrapper for all scib metrics from 
# https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/utils/util.py#L267
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT"
) -> Dict:
    
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal 
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:        
        log.warning(f"neighbors in adata.uns found \n {adata.uns['neighbors']} "
                    "\nto make sure the optimal clustering is calculated for the "
                    "correct embedding, removing neighbors from adata.uns."
                    "\nOverwriting calculation of neighbors with "
                    f"sc.pp.neighbors(adata, use_rep={embedding_key}).")
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        log.info("neighbors in adata.uns removed, new neighbors calculated: "
                 f"{adata.uns['neighbors']}")


    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()

    res_max, nmi_max, nmi_all = scib.metrics.clustering.opt_louvain(
            adata,
            label_key=label_key,
            cluster_key="cluster",
            use_rep=embedding_key,
            function=scib.metrics.nmi,
            plot=False,
            verbose=False,
            inplace=True,
            force=True,
    )
    
    results_dict["NMI_cluster/label"] = scib.metrics.nmi(
        adata, 
        "cluster",
        label_key,
        "arithmetic",
        nmi_dir=None
    )

    results_dict["ARI_cluster/label"] = scib.metrics.ari(
        adata, 
        "cluster", 
        label_key
    )

    results_dict["ASW_label"] = scib.metrics.silhouette(
        adata, 
        label_key, 
        embedding_key, 
        "euclidean"
    )   

    results_dict["graph_conn"] = scib.metrics.graph_connectivity(
        adata,
        label_key=label_key
    )
    

    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_batch"] = scib.metrics.silhouette(
            adata,
            batch_key,
            embedding_key,
            "euclidean"
        )

        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata, 
            batch_key,
            label_key, 
            embed=embedding_key, 
            metric="euclidean",
            return_all=False,
            verbose=False
        )

        results_dict["PCR_batch"] = scib.metrics.pcr(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            recompute_pca=True,
            n_comps=50,
            verbose=False
        )

    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )

    log.debug(
        "\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()])
    )

    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    return results_dict

def create_attention_mask_default(vecs: torch.Tensor) -> torch.Tensor:
    """
    Create an attention mask from a vector of positions of unknown genes. 
    Implementation of the mask described in the scGPT v2 preprint, fig S1A
    https://www.biorxiv.org/content/10.1101/2023.04.30.538439v2.full#F7

    Args:
        vecs (torch.Tensor): A bool tensor with position of unknown genes. 
                             shape (batch_size, seq_len)

    Returns:
        torch.Tensor: A bool attention mask. 
                      shape (batch_size, seq_len, seq_len)

    Examples:
    >>> create_attention_mask_default(torch.tensor([[0, 0, 1, 1],
                                                    [0, 0, 0, 1]]).bool())  
    tensor([[[False, False,  True,  True],
             [False, False,  True,  True],
             [False, False, False,  True],
             [False, False,  True, False]],

            [[False, False, False,  True],
             [False, False, False,  True],
             [False, False, False,  True],
             [False, False, False, False]]])

    """
    # check if vecs is a boolean tensor
    if not vecs.dtype == torch.bool:
        # check if vecs is 0 and 1 tensor
        if not torch.all(vecs.eq(0) | vecs.eq(1)):
            raise TypeError("vecs must be a boolean tensor")
        else:
            # convert to boolean tensor
            vecs = vecs.bool()

    # Use broadcasting to expand each vector into a square matrix
    attn_mask = vecs.unsqueeze(-1).repeat(1, 1, vecs.size(1))

    # Create a boolean mask for the diagonal
    diagonal_mask = ~torch.eye(vecs.size(1), device=vecs.device).bool()

    # Use the diagonal mask to set the diagonal of each square matrix to False
    attn_mask &= diagonal_mask

    # Transpose the last two dimensions to make each row of the original vector a column in the matrix
    attn_mask = attn_mask.transpose(-1, -2)

    return attn_mask

def create_attention_mask_modified(unknown_genes: torch.Tensor,
                                   cell_embedding_position: int = 0) -> torch.Tensor:
    """
    Create an attention mask from a vector of positions of unknown genes. 
    This is a modification of the mask described in the scGPT v2 preprint.
    With this attention mask, the cell embedding is always attended to,
    and the unknown genes are only attended to by the cell embedding.

    Args:
        unknown_genes (torch.Tensor): A bool or 0/1 tensor with position of unknown genes. 
                             shape (batch_size, seq_len)

    Returns:
        torch.Tensor: A bool attention mask. 
                      shape (batch_size, seq_len, seq_len)
    Examples:
    >>> create_attention_mask_modified(torch.tensor([[0, 0, 1, 1],
                                                     [0, 0, 0, 1]]).bool())  
    tensor([[[False, False,  True,  True],
             [False, False,  True,  True],
             [False, True, False,  True],
             [False, True,  True, False]],

            [[False, False, False,  True],
             [False, False, False,  True],
             [False, False, False,  True],
             [False, True, True, False]]])

    """
    # if not using GPU show warning
    if unknown_genes.device.type == 'cpu':
        log.warning('create_attention_mask_modified() is not optimized for CPU, '
                    'please use GPU for better performance.' 
                    f'Device: {unknown_genes.device.type} ')
    
    # # change vecs to 0 and 1 tensor
    # if unknown_genes.dtype == torch.bool:
    #     unknown_genes = unknown_genes.float()
    
    if torch.any(unknown_genes[:, cell_embedding_position].eq(True)): # eq(1)
        raise ValueError('Cell embedding position is unknown gene position.')
    
    attn_mask = create_attention_mask_default(unknown_genes)
    
    
    known_genes = ~unknown_genes
    # the cell embedding should always be attended to
    known_genes[:, cell_embedding_position] = False

    for i in range(len(unknown_genes)):
        attn_mask[i][unknown_genes[i].unsqueeze(-1) & known_genes[i].unsqueeze(0)] = True
            
    return attn_mask

def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
    mask_cell_embedding: bool = False,
    cell_emb_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.
        mask_cell_embedding (bool): Whether to mask the cell embedding, default to False.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crucial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    if not mask_cell_embedding:
        # sanity check that the first element is the cell embedding
        assert np.all(values[:, 0] == cell_emb_value)

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        if not mask_cell_embedding:
            # remove the first element, which is the cell embedding
            non_padding_idx = non_padding_idx[1:]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()

def mask_data(tokenized_data: Dict[str, torch.Tensor],
              mask_ratio: float = 0.15,
              pad_value: int = -2,
              mask_value: int = -1, 
              mask_cell_embedding: bool = False,
              cell_emb_value: int = 0) -> Dict[str, torch.Tensor]:
        """
        Mask the data.
        """
        # this will randomly mask all the values,
        # including or excluding the cell embedding based on the mask_cell_embedding flag
        masked_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio = mask_ratio,
            mask_value = mask_value,
            pad_value = pad_value,
            mask_cell_embedding = mask_cell_embedding,
            cell_emb_value = cell_emb_value
            )
        
        # this will mark the genes which values were masked for loss calculation
        gene_mask = torch.logical_and(tokenized_data["values"].ne(pad_value),
                                      masked_values.eq(mask_value))
    
        out_dict = {
            "gene_ids": tokenized_data["genes"],
            "values": masked_values,
            "target_values": tokenized_data["values"],
            "gene_mask": gene_mask
        }
        
        if "batch_labels" in tokenized_data.keys():
            out_dict["batch_labels"] = tokenized_data["batch_labels"]
        
        return out_dict

def calculate_losses(input: torch.Tensor,
                     output: torch.Tensor, 
                     masked_position: torch.Tensor,
                     non_padded_position: torch.Tensor, #TODO: add this!
                     skip_cell: bool = True,
                     methods: Union[List[str], str] = "all",
                     reduction_method = "mean",
                     cp: int = 0
                     ) -> Dict[str, float]:
    
    if reduction_method not in ["mean", "sum"]:
        msg = f"calculate_loss: reduction_method {reduction_method} is not supported"
        log.error(msg)
        raise ValueError(msg)
    
    implemented_methods = ["mse", "mre", "mae"]
    # check if methods equal to string all 
    if isinstance(methods, str) and methods.lower() == "all":
        methods = implemented_methods

    # if methods is not a list, make it a list    
    methods = [methods] if isinstance(methods, str) else methods

    # make sure methods are lower case
    methods = [method.lower() for method in methods]

    # check if element of methods is implemented
    methods_ = [method for method in methods if method not in implemented_methods]
    if len(methods_) == len(methods):
        msg = f"calculate_loss: methods {methods_} are not implemented"
        log.error(msg)
        raise ValueError(msg)
    
    if len(methods_) > 0:
        msg = f"calculate_loss: methods {methods_} are not implemented"
        log.warning(msg)

    # make sure masked_position is boolean
    if masked_position.dtype != torch.bool:
        # log.warning("evaluate_and_log: masked_position is not boolean")
        masked_position = masked_position.bool()

    if skip_cell:
        # check if cp is valid
        if cp >= input.shape[1]:
            msg = f"evaluate_and_log: cp {cp} is greater than input.shape[1] {input.shape[1]}"
            log.error(msg)
            raise ValueError(msg)
        # remove cp from input and output
        input = torch.concat((input[:,:cp], input[:,cp+1:]), dim=1)
        output = torch.concat((output[:,:cp], output[:,cp+1:]), dim=1)
        masked_position = torch.concat((masked_position[:,:cp], 
                                        masked_position[:,cp+1:]), dim=1)
        non_padded_position = torch.concat((non_padded_position[:,:cp],
                                            non_padded_position[:,cp+1:]), dim=1)
        
    results = dict()
    if "mse" in methods:
        # get the loss for masked values
        loss = F.mse_loss(torch.masked_select(output, masked_position).float(), 
                          torch.masked_select(input, masked_position).float(), 
                          reduction = reduction_method)
        
        # TODO: add masked_select on non padded!
        loss_all = F.mse_loss(torch.masked_select(output, non_padded_position).float(),
                              torch.masked_select(input, non_padded_position).float(), 
                              reduction = reduction_method)

        results["MSE"] = loss
        results["MSE_all"] = loss_all

    if "mre" in methods:
        loss = (torch.abs(torch.masked_select(output, masked_position) - 
                          torch.masked_select(input, masked_position)) /
                          (torch.masked_select(input, masked_position) + 1e-6))
        
        if reduction_method == "mean":
            loss = loss.mean()
        elif reduction_method == "sum":
            loss = loss.sum()

        loss_all = torch.abs((torch.masked_select(output, non_padded_position) -  
                              torch.masked_select(input, non_padded_position)) / 
                              (torch.masked_select(input, non_padded_position) + 1e-6))

        if reduction_method == "mean":
            loss_all = loss_all.mean()
        elif reduction_method == "sum":
            loss_all = loss_all.sum()

        results["MRE"] = loss
        results["MRE_all"] = loss_all

    if "mae" in methods:
        # get the loss for masked values
        loss = F.l1_loss(torch.masked_select(output, masked_position).float(), 
                         torch.masked_select(input, masked_position).float(), 
                         reduction = reduction_method)
        
        loss_all = F.l1_loss(torch.masked_select(output, non_padded_position).float(), 
                             torch.masked_select(input, non_padded_position).float(), 
                             reduction = reduction_method)
        
        results["MAE"] = loss
        results["MAE_all"] = loss_all
    
    return results

import torch

def permute_values(mat: torch.Tensor, 
                   pad_value: int = -2,
                   cell_embedding: bool = True) -> tuple:
    """
    Permute the data in a batch. The data is a 2D matrix with shape 
        (batch_size, seq_len). The non-padded values are permuted and
        pad values are kept at the end of the sequence
    Args:
        - mat: 2D matrix with shape (batch_size, seq_len)
        - pad_value: the value of the pad
        - cell_embedding: if True, the first position is a cell embedding
    Returns:
        - mat_perm: the permuted matrix
        - indx_perm: the indices of the permutation
    """
    if len(mat.shape) > 2:
        raise ValueError("mat should be 2D or 1D")
        
    # reshape the mat if needed so that the mat id 2D
    reshape = False
    if len(mat.shape) == 1:
        reshape = True
        mat = mat.unsqueeze(0)
    
    if cell_embedding:
        cemb = mat[:,0]
        mat = mat[:,1:]
    mat_perm = torch.empty_like(mat)
    indx_perm = torch.empty_like(mat)
    pad_masks = mat == pad_value

    for i in range(mat.shape[0]):
        non_pad = torch.where(~pad_masks[i])[0]
        pad = torch.where(pad_masks[i])[0]
        perm_non_pad = non_pad[torch.randperm(len(non_pad))] 
        perm = torch.cat([perm_non_pad, pad])
        indx_perm[i] = perm
        mat_perm[i] = mat[i][perm]
    
    if cell_embedding: # TODO: test not permuting cell_embedding!
        mat_perm = torch.cat([cemb.unsqueeze(1), mat_perm], dim=1)
        indx_perm = torch.cat((torch.zeros(indx_perm.shape[0], 1).to(indx_perm.device), 
                               indx_perm+1), dim=1)

    if reshape:
        mat_perm = mat_perm.view(-1)
        indx_perm = indx_perm.view(-1)

    # change indx_perm to int
    indx_perm = indx_perm.long()
    return mat_perm, indx_perm

def rearrange(mat: torch.Tensor, 
              indx: torch.Tensor) -> torch.Tensor:
    """
    Rearrange the rows of a 2D tensor A according to the indices indx
    Args:
        - mat: 2D tensor with shape (batch_size, seq_len)
        - indx: 2D tensor with shape (batch_size, seq_len)
    Returns:
        - mat_reordered: 2D tensor with shape (batch_size, seq_len)
    """
    batch_size = mat.shape[0]
    batch_indices = torch.arange(batch_size).view(-1, 1).to(indx.device)
    mat_reordered = mat[batch_indices, indx]
    return mat_reordered

def reverse_permute(mat: torch.Tensor,
                    indx: torch.Tensor) -> torch.Tensor:
    """
    Reverse the permutation of a 2D tensor A according to the indices indx
    Args:
        - mat: 2D tensor with shape (batch_size, seq_len)
        - indx: 2D tensor with shape (batch_size, seq_len)
    Returns:
        - Sorted 2D tensor with shape (batch_size, seq_len)
    """
    dims = torch.arange(indx.size(0)).reshape(-1, 1)
    return mat[dims, indx.argsort()]
