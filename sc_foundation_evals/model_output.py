## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os
from typing import List, Optional, Union, Dict

from . import utils
from . import data

from .geneformer_forward import Geneformer_instance
from .scgpt_forward import scGPT_instance

from .helpers.custom_logging import log

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import PyComplexHeatmap as pych

import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

def check_attributes(object, attribute: str):
    """
    check if the attribute is in the object
    """
    if not hasattr(object, attribute):
        msg = f"{attribute} is not an attribute of {object}"
        log.error(msg)
        raise ValueError(msg)

def save_tensor_as_csv(tensor: torch.Tensor,
                       output_dir: str,
                       filename: str) -> None:
    """
    save a tensor as csv
    """
    if not os.path.exists(output_dir):
        log.warning(f"Creating the output directory {output_dir}")
        os.makedirs(output_dir)
    
    if not isinstance(tensor, torch.Tensor):
        msg = "tensor must be a torch.Tensor"
        log.error(msg)
        raise ValueError(msg)
    
    tensor_df = pd.DataFrame(tensor.numpy(),
                             index = range(tensor.shape[0]),
                             columns = range(tensor.shape[1]))
    
    tensor_df.to_csv(os.path.join(output_dir, filename))


def columnwise_correlation(tensor1, tensor2):
    # Calculate means along each column
    mean1 = torch.mean(tensor1, dim=0)
    mean2 = torch.mean(tensor2, dim=0)
    
    # Subtract means from each element in the column (broadcasted automatically)
    diff1 = tensor1 - mean1
    diff2 = tensor2 - mean2
    
    # Calculate Pearson correlation for each pair of columns
    numerator = torch.sum(diff1 * diff2, dim=0)
    denominator = torch.sqrt(torch.sum(diff1 ** 2, dim=0) * torch.sum(diff2 ** 2, dim=0))
    
    # Prevent division by zero and calculate correlation
    correlation = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(denominator))
    
    return correlation


class GeneExprPredEval():
    def __init__(self,
                 model_instance: Union[scGPT_instance, 
                                       Geneformer_instance],
                 data: Optional[data.InputData] = None,
                 embedding_key: Optional[Union[str, List[str]]] = "mlm_output",
                 output_dir: Optional[str] = None,
                 log_wandb: bool = False) -> None:

        # test if scGPT_instance is a valid instance
        if not isinstance(model_instance, 
                          (scGPT_instance, Geneformer_instance)):
            msg = "model_instance must be an instance of scGPT_instance"
            log.error(msg)
            raise ValueError(msg)
        
        # if wandb set to true and not initialized, throw error
        if log_wandb and not model_instance._wandb:
            msg = "wandb is not initialized in model_instance"
            log.error(msg)
            raise ValueError(msg)
        
        self._wandb = model_instance._wandb
        
        self.eval_instance = model_instance
        self.embedding_key = embedding_key

        self.model_type = ("scgpt" 
                           if isinstance(self.eval_instance, scGPT_instance) 
                           else "geneformer")
        if self.model_type == "scgpt":
            self.data = data
            embedding_key = ([embedding_key] if isinstance(embedding_key, str) 
                            else embedding_key)
            # check if embedding_key is valid
            for key in embedding_key:
                check_attributes(self.eval_instance, key)
        
        elif self.model_type == "geneformer":
            # check if rankings are saved
            rankings = ["input_rankings", "output_rankings"]
            for ranking in rankings:
                check_attributes(self.eval_instance, ranking)


        if output_dir is not None:
            # if output dir is provided, use it
            self.output_dir = output_dir
            # check if output_dir exists
            if not os.path.exists(self.output_dir):
                log.warning(f"Creating the output directory {self.output_dir}")
                os.makedirs(self.output_dir)
        else:
            # use the same output_dir as the model_instance
            self.output_dir = self.eval_instance.output_dir



    def _evaluate_Geneformer(self, 
                             n_cells: Optional[int] = 1000,
                             return_all: bool = False,
                             save_rankings: bool = False) -> None:

        n_all_cells = len(self.eval_instance.input_rankings)

        if n_cells is None:
            n_cells = n_all_cells
            rand_cells = np.arange(n_all_cells)

        elif n_cells > n_all_cells:
            msg = (f"n_cells {n_cells} is larger than the number of cells in the "
                   f"dataset ({n_all_cells}); setting to the maximum")
            log.warning(msg)
            n_cells = n_all_cells

        elif n_cells < 1:
            msg = "n_cells must be greater than 0"
            log.error(msg)
            raise ValueError(msg)
        
        log.debug(f"Extracting output from {n_cells} cells")
        
        if n_cells < n_all_cells:
            rand_cells = np.random.choice(range(n_all_cells), 
                                          n_cells, replace = False)
            in_rankings = [self.eval_instance.input_rankings[i] 
                           for i in rand_cells]
            out_rankings = [self.eval_instance.output_rankings[i] 
                            for i in rand_cells]

        else:
            in_rankings = self.eval_instance.input_rankings
            out_rankings = self.eval_instance.output_rankings

        # Find unique tokens across all arrays
        unique_tokens = np.unique(np.concatenate(in_rankings + out_rankings))
        # Number of unique tokens and cells
        n_tokens = len(unique_tokens)
        
        # Initialize tensors with zeros (will fill in actual values later)
        in_ranks = np.zeros((n_tokens, n_cells))
        out_ranks = np.zeros((n_tokens, n_cells))

        log.debug("Populating tensors with actual positions")
        # Populate tensors with actual positions
        for j in range(n_cells):
            # Get the max positions for the current cell
            max_pos_in = len(in_rankings[j]) + 1

            # only take a sequence as long as the input
            out_rankings[j] = out_rankings[j][:len(in_rankings[j])]
            
            max_pos_out = len(out_rankings[j]) + 1
            
            # Initialize to max position specific to the cell
            in_ranks[:, j] = max_pos_in
            out_ranks[:, j] = max_pos_out
            
            for i, token in enumerate(unique_tokens):
                pos_in_in = np.where(in_rankings[j] == token)[0]
                # the question here is wether the token apears multiple times
                # run a notebook and look at the outputs
                pos_in_out = np.where(out_rankings[j] == token)[0]

                if pos_in_in.size > 0:
                    in_ranks[i, j] = pos_in_in[0] + 1  # 1-based index
                if pos_in_out.size > 0:
                    out_ranks[i, j] = np.rint(np.mean(pos_in_out)) + 1  # 1-based index
                    
        log.debug("Finished populating tensors")
        # Convert to PyTorch tensors
        in_ranks = torch.tensor(in_ranks, dtype=torch.int)
        out_ranks = torch.tensor(out_ranks, dtype=torch.int)

        if save_rankings:
            # save tensor as csv
            save_tensor_as_csv(in_ranks,
                               self.output_dir,
                               "input_rankings.csv.gz")
            save_tensor_as_csv(out_ranks,
                               self.output_dir,
                               "out_rankings.csv.gz")

        # Get the ranking  
        in_ranks = torch.max(in_ranks, dim=0)[0].expand_as(in_ranks) - in_ranks.float()
        in_ranks = in_ranks / torch.max(in_ranks, dim=0)[0].expand_as(in_ranks)
        out_ranks = torch.max(out_ranks, dim=0)[0].expand_as(out_ranks) - out_ranks.float()
        out_ranks = out_ranks / torch.max(out_ranks, dim=0)[0].expand_as(out_ranks)
        
        # calculate mean ranks across cells
        mean_ranks = torch.mean(in_ranks, dim = 1)
        mean_ranks = mean_ranks.repeat(in_ranks.shape[1], 1).T

        self.in_ranks = in_ranks
        self.out_ranks = out_ranks
        self.mean_ranks = mean_ranks

        if save_rankings:
            save_tensor_as_csv(in_ranks,
                               self.output_dir,
                               "input_scaled_rankings.csv.gz")
            save_tensor_as_csv(out_ranks,
                               self.output_dir,
                               "output_scaled_rankings.csv.gz")
            save_tensor_as_csv(mean_ranks,
                               self.output_dir,
                               "mean_rankings.csv.gz")

        corrs = columnwise_correlation(in_ranks, out_ranks)
        mean_corrs = columnwise_correlation(in_ranks, mean_ranks)
        
        metrics_df = pd.DataFrame({"corr": [corrs.mean().item(), 
                                            mean_corrs.mean().item()]},
                            index = ["geneformer_out", "mean_rankiing"])
        
        # save the metrics
        metrics_df.to_csv(os.path.join(self.output_dir, 
                                       "gene_embeddings_metrics.csv"))
        # log the metrics to wandb
        if self._wandb:
            self._wandb.log({"gene_embeddings_metrics": 
                             self._wandb.Table(dataframe = metrics_df)})
            
        if return_all:
            return metrics_df, corrs, mean_corrs

        return metrics_df


    def _evaluate_scGPT(self,
                        remove_cell_embedding: bool = True,
                        include_zero_genes: bool = False) -> None:
        # take the embeddings and input and calculate the metrics using 
        # the loss function        
        metrics_df = pd.DataFrame()
        input_values = self.eval_instance.tokenized_data['values']

        if include_zero_genes:
            masked_values_ = torch.full_like(input_values, 
                                             True, 
                                             dtype=torch.bool)
        else:
            masked_values_ = input_values > 0

        non_padded_values = input_values.ne(self.eval_instance.model_config['pad_value'])
        
        for emb_key in self.embedding_key:
            embedding = getattr(self.eval_instance, emb_key)
            # check if embedding is a tensor
            if not isinstance(embedding, torch.Tensor):
                msg = f"{emb_key} is not a tensor"
                log.warning(msg)
                embedding = torch.tensor(embedding)
            
            metrics = utils.calculate_losses(embedding, 
                                             input_values, 
                                             masked_values_,
                                             non_padded_values,
                                             skip_cell = remove_cell_embedding)
            metrics = {key: value.item() for key, value in metrics.items()}
            metrics_df = pd.concat([metrics_df,
                                    pd.DataFrame(metrics, index = [emb_key])])
        
        # add the reference if the prediction would have been the mean
        mean = np.mean(input_values[masked_values_].detach().cpu().numpy())
        mean_values = torch.full_like(input_values, mean)
        metrics = utils.calculate_losses(mean_values, 
                                         input_values, 
                                         masked_values_,
                                         non_padded_values,
                                         skip_cell = remove_cell_embedding)
        metrics = {key: value.item() for key, value in metrics.items()}
        metrics_df = pd.concat([metrics_df,
                                pd.DataFrame(metrics, index = ["mean"])])

        # save the metrics
        metrics_df.to_csv(os.path.join(self.output_dir, 
                                       "gene_embeddings_metrics.csv"))
        
        # log the metrics to wandb
        if self._wandb:
            self._wandb.log({"gene_embeddings_metrics": 
                             self._wandb.Table(dataframe = metrics_df)})
            
        return metrics_df
    

    def evaluate(self,
                 remove_cell_embedding: Optional[bool] = True,
                 include_zero_genes: Optional[bool] = False,
                 n_cells: Optional[int] = 1000,
                 return_all: Optional[bool] = False,
                 save_rankings: Optional[bool] = False) -> None:
        """
        Evaluate the model output
        """

        if self.model_type == "scgpt":
            return self._evaluate_scGPT(remove_cell_embedding = remove_cell_embedding,
                                        include_zero_genes = include_zero_genes)
        
        elif self.model_type == "geneformer":
            return self._evaluate_Geneformer(n_cells = n_cells,
                                             return_all = return_all,
                                             save_rankings = save_rankings)
            
           
    
    def _visualize_Geneformer(self, 
                              n_cells: Optional[int] = 1000, 
                              cmap = "Blues") -> plt.figure:

        n_all_available_cells = self.in_ranks.shape[1]
        n_cells = n_all_available_cells if n_cells is None else n_cells

        if n_cells < n_all_available_cells:
            msg = f"Subsetting to {n_cells} cells"
            log.info(msg)
            # select random cells
            rand_cells = np.random.choice(range(n_all_available_cells),
                                            n_cells, replace=False)
            in_ranks = self.in_ranks[:, rand_cells].flatten()
            out_ranks = self.out_ranks[:, rand_cells].flatten()
            mean_ranks = self.mean_ranks[:, rand_cells].flatten()
        elif n_cells < 1:
            msg = f"n_cells must be greater than 0; provided: {n_cells}"
            log.error(msg)
            raise ValueError(msg)
        else:    
            in_ranks = self.in_ranks.flatten()
            out_ranks = self.out_ranks.flatten()
            mean_ranks = self.mean_ranks.flatten()

        # remove the genes absent in input and output ranks
        subset_non_zero = torch.logical_or(in_ranks > 0, out_ranks > 0)
        in_ranks = in_ranks[subset_non_zero]
        out_ranks = out_ranks[subset_non_zero]
        mean_ranks = mean_ranks[subset_non_zero]

        # set seaborn style
        sns.set_style("white")


        # get two plots side by side
        fig, (ax1, ax2) = plt.subplots(ncols = 2, 
                                       sharey = True, 
                                       tight_layout = True, 
                                       figsize = (10, 5))
        ax1.set_xlabel("Input ranks")
        ax1.set_ylabel("Geneformer reconstructed ranks")
        ax2.set_xlabel("Input ranks")
        ax2.set_ylabel("Mean ranks")

        sns.kdeplot(x = in_ranks, 
                    y = out_ranks, 
                    cmap = cmap,
                    fill = True, 
                    shade = True,
                    thresh = 0, ax = ax1)
        sns.kdeplot(x = in_ranks, 
                    y = mean_ranks, 
                    cmap = cmap,  
                    shade = True,
                    fill = True, thresh = 0, ax = ax2)
        
        plt.suptitle("Correlation between input and reconstructed and mean rankings")

        save_path = os.path.join(self.output_dir, 
                                "input_and_outputs_kde.png")
        
        plt.savefig(save_path, bbox_inches='tight')
        if self._wandb:
            self._wandb.log({"input_and_outputs_kde": 
                             self._wandb.Image(save_path)})
        return plt
        

    def _visualize_scGPT(self, 
                         label_key: str = "cell_type",
                         skip_cell: bool = True) -> plt.figure:
        # get the cell embedding 
        input_values = self.eval_instance.tokenized_data['values']
        mlm_output = self.eval_instance.mlm_output
        mvc_output = self.eval_instance.mvc_output
        
        # arrange columns by mean value
        mean_value = np.mean(input_values.detach().cpu().numpy(), axis = 0)
        sort_indices = np.argsort(-mean_value)
        input_values = input_values[:, sort_indices]
        mlm_output = mlm_output[:, sort_indices]
        mvc_output = mvc_output[:, sort_indices]
        
        # get the random cells
        rand_indices = np.random.choice(np.arange(input_values.shape[0]),
                                        size = 100, replace = False)
        input_values = input_values[rand_indices, :]
        mlm_output = mlm_output[rand_indices, :]
        mvc_output = mvc_output[rand_indices, :]
        cell_type_annot = self.data.adata.obs[label_key].values[rand_indices]
        cm_values = self.plot_heatmap(input_values,
                                      label = "Input values",
                                      output_dir = self.output_dir,
                                      plot_name = "input_values_heatmap",
                                      annot = False,
                                      cell_type_annotation = cell_type_annot,
                                      return_fig = True,
                                      plot_fig = False,
                                      skip_cell = skip_cell)
        cm_mlm_output = self.plot_heatmap(mlm_output,
                                     label = "MLM output",
                                     # cmap = "Greens",
                                     output_dir = self.output_dir,
                                     plot_name = "mlm_output_heatmap",
                                     annot = False,
                                     return_fig = True,
                                     plot_fig = False,
                                     skip_cell = skip_cell)
        cm_mvc_output = self.plot_heatmap(mvc_output,
                                     label = "MVC output",
                                     # cmap = "Reds",
                                     output_dir = self.output_dir,
                                     plot_name = "mvc_output_heatmap",
                                     annot = False,
                                     return_fig = True, 
                                     plot_fig = False,
                                     skip_cell = skip_cell)
        
        with plt.style.context('ggplot'):
            plt.figure(figsize = (25, 7))
            ax, legend_axes = pych.composite(cmlist = [cm_values, 
                                                       cm_mlm_output, 
                                                       cm_mvc_output], 
                                             main = 0, 
                                             legend_hpad = 10, 
                                             legend_gap = 17)
        
            cm_values.ax.set_title("Input bins")
            cm_mlm_output.ax.set_title("MLM output")
            cm_mvc_output.ax.set_title("MVC output")
            model_run = self.eval_instance.model_run
            ax.set_title((f"Comparison of input and output of the {model_run} "
                          "scGPT model"),
                        y=1.05, fontdict={'fontweight':'bold'})
            save_path = os.path.join(self.output_dir, 
                                     "input_and_outputs_heatmap.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            if self._wandb:
                self._wandb.log({"input_and_outputs_heatmap": 
                                 self._wandb.Image(save_path)})
        return plt

    def visualize(self,
                  n_cells: Optional[int] = 1000,
                  label_key: Optional[str] = "cell_type",
                  skip_cell: Optional[bool] = True, **kwargs) -> None:
        
        if self.model_type == "scgpt":
            self._visualize_scGPT(label_key = label_key,
                                  skip_cell = skip_cell)
        
        elif self.model_type == "geneformer":
            self._visualize_Geneformer(n_cells = n_cells, **kwargs)


    @staticmethod
    def plot_heatmap(mat: Union[np.ndarray, torch.Tensor],
                     label = "MVC_output", 
                     cmap = "Blues", 
                     annot: bool = True,
                     cell_type_annotation: Optional[List] = None,
                     return_fig: bool = False,
                     plot_fig: bool = True,
                     output_dir: Optional[str] = None,
                     plot_name: str = "heatmap",
                     skip_cell: bool = True) -> Optional[pych.ClusterMapPlotter]:

        # if tensor detach and move to cpu and numpy
        if type(mat) == torch.Tensor:
            mat = mat.detach().cpu().numpy()

        if annot and cell_type_annotation is None:
            msg = "cell_type_annotation is required for annot = True"
            log.error(msg)
            raise ValueError(msg)
        
        if not return_fig and output_dir is None:
            msg = "output_dir is required if return_fig is False"
            log.error(msg)
            raise ValueError(msg)

        if skip_cell:
            mat = mat[:, 1:]
        
        
        mean_value_ = np.mean(mat, axis = 0)
        mean_value_ = pd.Series(mean_value_, 
                                index = np.arange(mean_value_.shape[0]))
        rand_indices = list(range(mat.shape[0]))
        if mat.shape[0] > 100:
            msg = (f"Matrix is too large to plot ({mat.shape}), "
                   "subsetting to 100 cells")
            log.warning(msg)
            rand_indices = np.random.choice(mat.shape[0], 100, replace=False)
            mat = mat[rand_indices, :]
            if annot:
                cell_type_annotation = cell_type_annotation[rand_indices]
        
        if annot:
            cell_types = np.unique(cell_type_annotation)
            cell_type_colors = dict(zip(cell_types, 
                                        plt.cm.viridis(np.linspace(0, 1, 
                                                                   len(cell_types)))))
            # if cell annotation is not DataFrame or Series make it into one
            if (type(cell_type_annotation) != pd.DataFrame and 
                type(cell_type_annotation) != pd.Series):
                msg = (f"cell_type_annotation is {type(cell_type_annotation)} "
                       "but must be a pandas DataFrame or Series; changing")
                log.warning(msg)
                cell_type_annotation = pd.Series(cell_type_annotation, 
                                                 index = rand_indices)
            row_ha = pych.HeatmapAnnotation(Cell = pych.anno_simple(cell_type_annotation,
                                                          colors = cell_type_colors,
                                                          legend = True),
                                       legend_gap = 5, 
                                       hgap = 0.5, 
                                       label_side = 'bottom',
                                       axis = 0)

        log.debug(f"mean_value_ shape: {mean_value_.shape}")
        log.debug(f"mat shape: {mat.shape}")
        
        plot_df = pd.DataFrame(mat, 
                               index = range(mat.shape[0]), 
                               columns = range(mat.shape[1])) 

        col_ha = pych.HeatmapAnnotation(Bin = pych.anno_barplot(mean_value_,
                                                                legend = False),
                                        axis = 1, plot_legend = False)
        
        plt.figure(figsize = (10, 10))    
        vmin = round(np.nanmin(mat[mat != -np.inf]), 0)
        vmax = round(np.nanmax(mat[mat != np.inf]), 0)
        
        if annot:
            cmp = pych.ClusterMapPlotter(data = plot_df, 
                                    top_annotation = col_ha, 
                                    left_annotation = row_ha, 
                                    plot = plot_fig,
                                    plot_legend = True, 
                                    legend_hpad = 2, 
                                    legend_vpad = 10,
                                    label = label, 
                                    legend_gap = 17,
                                    cmap = cmap,
                                    row_cluster = False, 
                                    col_cluster = False,
                                    verbose = False,
                                    legend_kws = {"vmin": vmin, 
                                                  "vmax": vmax})
        else:
            cmp = pych.ClusterMapPlotter(data = plot_df, 
                                    top_annotation = col_ha, 
                                    left_annotation = None, 
                                    plot = plot_fig,
                                    plot_legend = True, 
                                    legend_hpad = 2, 
                                    legend_vpad = 10,
                                    label = label, 
                                    legend_gap = 17,
                                    cmap = cmap,
                                    row_cluster = False, 
                                    col_cluster = False,
                                    verbose = False,
                                    legend_kws = {"vmin": vmin, 
                                                  "vmax": vmax})
                                    
        if output_dir is not None:
            # save the plot
            plt.savefig(os.path.join(output_dir, 
                                    f"{plot_name}.png"))

        if return_fig:
            return cmp
    
    @staticmethod
    def plot_boxplot():
        pass