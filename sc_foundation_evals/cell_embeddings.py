## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import os
from typing import List, Optional, Tuple, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import seaborn as sns
import scanpy as sc

from .helpers import umap
from .helpers.custom_logging import log

from . import data, utils
from .geneformer_forward import Geneformer_instance
from .scgpt_forward import scGPT_instance

class CellEmbeddingsEval():
    def __init__(self,
                 model_instance: Union[scGPT_instance, 
                                       Geneformer_instance],
                 data: data.InputData,
                 label_key: Union[str, List[str]] = "cell_type",
                 batch_key: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 log_wandb: bool = False) -> None:
        
        # test if model_instance is an instance of scGPT_instance or Geneformer_instance
        if not isinstance(model_instance, 
                          (scGPT_instance, Geneformer_instance)):
            msg = ("scgpt_instance must be an instance of "
                   "scGPT_instance or Geneformer_instance")
            log.error(msg)
            raise ValueError(msg)
        
        # test if instance is properly processed
        if not hasattr(model_instance, "cell_embeddings"):
            msg = "Cell embeddings need to be extracted first"
            log.error(msg)
            raise ValueError(msg)

        # if wandb set to true and not initialized, throw error
        if log_wandb and not model_instance._wandb:
            msg = "wandb is not initialized in model_instance"
            log.error(msg)
            raise ValueError(msg)
        
        self._wandb = model_instance._wandb

        self.eval_instance = model_instance
        self.data = data

        if batch_key is not None:
            if batch_key not in self.data.adata.obs.columns:
                msg = f"batch_key {batch_key} not found in adata.obs"
                log.error(msg)
                raise ValueError(msg)
            else:
                self.batch_key = batch_key
        else:
            try:
                self.batch_key = self.data.batch_str_col
            except AttributeError:
                msg = "batch_key not provided and not found in data object"
                log.error(msg)
                raise ValueError(msg)

        if output_dir is not None:
            # if output dir is provided, use it
            self.output_dir = output_dir
            # check if output_dir exists
            if not os.path.exists(self.output_dir):
                log.warning(f"Creating the output directory {self.output_dir}")
                os.makedirs(self.output_dir)
        else:
            # use the same output_dir as the scgpt_instance
            self.output_dir = self.eval_instance.output_dir

        # if label_key is string, convert to list
        if isinstance(label_key, str):
            label_key = [label_key]
        self.label_key = label_key

        # make sure that each label exists and is categorical in adata.obs
        for label in self.label_key:
            if label not in self.data.adata.obs.columns:
                msg = f"Label {label} not found in adata.obs"
                log.error(msg)
                raise ValueError(msg)
            self.data.adata.obs[label] = self.data.adata.obs[label].astype("category")
        
    def evaluate(self, 
                 embedding_key: str = "X_scGPT",
                 n_cells: int = 7500) -> pd.DataFrame:
        
        adata_ = self.data.adata.copy()
        
        # if adata_ too big, take a subset
        if adata_.n_obs > n_cells:
            log.warning(f"adata_ has {adata_.n_obs} cells. "
                        f"Taking a subset of {n_cells} cells.")
            sc.pp.subsample(adata_, n_obs = n_cells, copy = False)

        met_df = pd.DataFrame(columns = ["metric", "label", "value"])

        # get unique values in self.label_key preserving the order
        label_cols = [x for i, x in enumerate(self.label_key) 
                      if x not in self.label_key[:i]]
        # remove label columns that are not in adata_.obs
        label_cols = [x for x in label_cols if x in adata_.obs.columns]

        if len(label_cols) == 0:
            msg = f"No label columns {self.label_key} found in adata.obs"
            log.error(msg)
            raise ValueError(msg)
        
        # check if the embeddings are in adata
        if embedding_key not in adata_.obsm.keys():
            msg = f"Embeddings {embedding_key} not found in adata.obsm"
            log.error(msg)
            raise ValueError(msg)
        
        for label in label_cols:
            log.debug(f"Computing metrics for {label}")
           
            metrics = utils.eval_scib_metrics(adata_,
                                              batch_key = self.batch_key, 
                                              label_key = label,
                                              embedding_key = embedding_key)
            for metric in metrics.keys():
                log.debug(f"{metric} for {label}: {metrics[metric]}")

                # log to wandb if initialized
                if self._wandb:
                    self._wandb.log({f"{embedding_key}/{label}/{metric}": metrics[metric]})
                
                # add row to the dataframe
                met_df.loc[len(met_df)] = [metric, label, metrics[metric]]
        
        met_df.to_csv(os.path.join(self.output_dir, 
                                   f"{embedding_key}__metrics.csv"), 
                      index = False)
        
        if self._wandb:
            wandb_df = self._wandb.Table(data = met_df)
            self._wandb.log({f"{embedding_key}/metrics": wandb_df})
        return met_df
    
    def create_original_umap(self,
                             out_emb: str = "X_umap_input") -> None:
        
        sc.pp.neighbors(self.data.adata)
        temp = sc.tl.umap(self.data.adata, min_dist = 0.3, copy=True)
        self.data.adata.obsm[out_emb] = temp.obsm["X_umap"].copy()

    # TODO: this should be a more generic function that can plot any embedding
    def visualize(self, 
                  embedding_key: str = "X_scGPT",
                  return_fig: bool = False,
                  plot_size: Tuple[float, float] = (9, 7),
                  plot_title: Optional[str] = None,
                  plot_type: [List, str] = "simple",
                  n_cells: int = 7500
                  ) -> Optional[Dict[str, plt.figure]]:
        
        raw_emb = "X_umap_input"

        if embedding_key == raw_emb:
            # if the umap_raw embedding is used, create it first
            self.create_original_umap(out_emb = embedding_key)

        # if adata already has a umap embedding warn that it will be overwritten
        if "X_umap" in self.data.adata.obsm.keys():
            old_umap_name = "X_umap_old"
            log.warning(f"Copying existing UMAP embedding to {old_umap_name} "
                        "and overwriting X_umap.")
            self.data.adata.obsm[old_umap_name] = self.data.adata.obsm["X_umap"].copy()
        
        # check if the embeddings are in adata
        if embedding_key not in self.data.adata.obsm.keys():
            msg = f"Embeddings {embedding_key} not found in adata."
            log.error(msg)
            raise ValueError(msg)

        # if embedding_key contains the string umap, do not compute umap again
        if embedding_key != raw_emb:
            # compute umap embeddings
            sc.pp.neighbors(self.data.adata, use_rep = embedding_key)
            sc.tl.umap(self.data.adata, min_dist = 0.3)
         
        adata_ = self.data.adata.copy()
        # if adata_ too big, take a subset
        if adata_.n_obs > n_cells:
            log.warning(f"adata_ has {adata_.n_obs} cells. "
                        f"Taking a subset of {n_cells} cells.")
            sc.pp.subsample(adata_, n_obs = n_cells, copy = False)
            # save the subsetted adata.obs
            adata_.obs.to_csv(os.path.join(self.output_dir,
                                             "adata_obs_subset.csv"))


        
        # make sure plot size is a tuple of numbers
        try: 
            w, h = plot_size
            if not isinstance(h, (int, float)) or not isinstance(w, (int, float)):
                msg = f"Height (h = {h}) or width (w = {w}) not valid."
                log.error(msg)
                raise TypeError(msg)
        except TypeError:
            msg = f"Plot size {plot_size} is not a tuple of numbers."
            log.error(msg)
            raise TypeError(msg)
            
        # get unique values in self.label_key preserving the order
        label_cols = self.label_key + [self.batch_key]
        label_cols = [x for i, x in enumerate(label_cols) 
                      if x not in label_cols[:i]]
        # remove label columns that are not in adata_.obs
        label_cols = [x for x in label_cols 
                      if x in self.data.adata.obs.columns]
        
        if len(label_cols) == 0:
            msg = f"No label columns {self.label_key} found in adata.obs"
            log.error(msg)
            raise ValueError(msg)
        
        # set the colors for the labels
        labels = dict()
        labels_colors = dict()
        palettes = ['viridis', 'inferno',
                    'mako', 'rocket',
                    'tab20', 'colorblind',  
                    'tab20b', 'tab20c']
        
        if len(label_cols) > len(palettes):
            log.warning("More labels than palettes. Adding random colors.")
            palettes = palettes + ["random"] * (len(label_cols) - len(palettes))

        # creating palettes for the labels
        for i, label in enumerate(label_cols):
            labels[label] = self.data.adata.obs[label].unique()
            if len(labels[label]) > 10:
                log.warning(f"More than 10 labels for {label}."
                            f"The plots might be hard to read.")
            labels_colors[label] = dict(zip(labels[label],
                                        umap.generate_pallette(n = len(labels[label]),
                                                                        cmap = palettes[i])))
        
        
        
        figs = {}

        # if plot_type a string, convert to list
        if isinstance(plot_type, str):
            plot_type = [plot_type]
        
        plot_type = [x.lower() for x in plot_type]
        # get unique values in plot_type
        plot_type = [x for i, x in enumerate(plot_type) 
                     if x not in plot_type[:i]]
        old_plot_type = plot_type
        # check if plot_type is valid
        valid_plot_types = ["simple", "wide", "scanpy"]
        
        # create a subset of plot_type that is valid
        plot_type = [x for x in plot_type if x in valid_plot_types]
        if len(plot_type) == 0:
            msg = f"Plot type {plot_type} is not valid. Valid plot types are {valid_plot_types}"
            log.error(msg)
            raise ValueError(msg)
        
        # print a warning if plot_type is not valid
        if len(plot_type) < len(old_plot_type):
            log.warning(f"Some plot type(s) {old_plot_type} is not valid. "
                        f"Valid plot types are {valid_plot_types}. "
                        f"Plotting only {plot_type}")


        plt_emb = "X_umap" if embedding_key != raw_emb else embedding_key

        plot_title = (plot_title 
                      if plot_title is not None 
                      else "UMAP of the cell embeddings")

        if "simple" in plot_type:
            fig, axs = plt.subplots(ncols = len(label_cols), 
                                    figsize = (len(label_cols) * w, h),
                                    squeeze = False)
        
            axs = axs.flatten()
            
            # basic plotting, problematic: size of the points
            embedding = self.data.adata.obsm[plt_emb]
            for i, label in enumerate(label_cols): 
                log.debug(f"Plotting the embeddings for {label}")
                # remove axis and grid from the plot
                axs[i].axis('off')
                # plot umap embeddings, add color by cell type
                axs[i].scatter(embedding[:, 0], embedding[:, 1],
                            # make points smaller
                            s = 0.5, 
                            c = [labels_colors[label][x] for x 
                                in self.data.adata.obs[label]])
                legend_handles = [axs[i].plot([], [], 
                                            marker = "o", ls = "", 
                                            color = c, label = l)[0]
                                            for l, c in labels_colors[label].items()]
                axs[i].legend(handles = legend_handles, 
                            bbox_to_anchor = (1.05, 1), 
                            loc = 'upper left')
                
                # Add a title to the plot
                axs[i].title.set_text(f"{label}")

            fig.suptitle(plot_title, fontsize = 16)
            fig.tight_layout()
            fig.subplots_adjust(top = 0.85)

            fig_savefig = os.path.join(self.output_dir, 
                                    f"umap__{embedding_key}.png")
            fig.savefig(fig_savefig)

            # if wandb initialized, log the figure
            if self._wandb:
                self._wandb.log({f"umap__{embedding_key}": self._wandb.Image(fig_savefig)})

            if return_fig:
                figs["umap"] = fig

        # wide plotting
        if "wide" in plot_type:
            df = pd.DataFrame(self.data.adata.obsm[plt_emb], 
                              columns = ["umap_1", "umap_2"])
            for i, label in enumerate(label_cols):
                if self.data.adata.obs[label].unique().shape[0] <= 10:
                    df[label] = self.data.adata.obs[label].tolist()
                    wide_plot = sns.relplot(data = df, 
                                            col = label,
                                            x = "umap_1", 
                                            y = "umap_2",
                                            hue = label, 
                                            style = label, 
                                            legend = "full", 
                                            palette = palettes[i])
                    # switch off axes
                    for axes in wide_plot.axes.flat:
                        axes.set_axis_off()
                    sns.move_legend(wide_plot, "upper left", bbox_to_anchor=(1, 1))
                    wide_plot.fig.suptitle(plot_title, fontsize = 16)
                    wide_plot.fig.tight_layout()
                    wide_plot.fig.subplots_adjust(top = 0.85)

                    wide_plot_savefig = os.path.join(self.output_dir,
                                                    f"umap_wide__{embedding_key}_{label}.png")
                    wide_plot.savefig(wide_plot_savefig)
                    
                    # if wandb initialized, log the figure 
                    if self._wandb:
                        self._wandb.log({f"umap_wide__{embedding_key}_{label}": self._wandb.Image(wide_plot_savefig)})
                    if return_fig:
                        figs[label] = wide_plot
                else:
                    msg = f"More than 10 labels for {label}. Skipping wide plot."
                    log.warning(msg)

                
        if "scanpy" in plot_type:
            # scanpy plotting
            labels_colors_flat = {k: v for d in labels_colors 
                                for k, v in labels_colors[d].items()}
            if embedding_key == raw_emb:
                # TODO: this needs rewriting
                adata_temp__ = self.data.adata.copy()
                adata_temp__.obsm["X_umap"] = self.data.adata.obsm[raw_emb].copy()
                fig2 = sc.pl.umap(adata_temp__, 
                                color = label_cols,
                                add_outline = True,
                                layer = plt_emb,
                                legend_loc = 'on data',
                                palette = labels_colors_flat,
                                return_fig = True)
                # remove the temporary adata
                del adata_temp__
            else:  
                fig2 = sc.pl.umap(self.data.adata, 
                                color = label_cols,
                                add_outline = True,
                                layer = plt_emb,
                                legend_loc = 'on data',
                                palette = labels_colors_flat,
                                return_fig = True)
            fig2.suptitle(plot_title, fontsize = 16)
            fig2.tight_layout()
            fig2.subplots_adjust(top = 0.85)
        
            fig2_savefig = os.path.join(self.output_dir, 
                                        f"umap_scanpy__{embedding_key}.png")
            fig2.savefig(fig2_savefig)
            
            # if wandb initialized, log the figure
            if self._wandb:
                self._wandb.log({f"umap_scanpy/{embedding_key}": self._wandb.Image(fig2_savefig)})

            if return_fig:
                figs["umap_scanpy"] = fig2

        
        if return_fig:
            return figs
