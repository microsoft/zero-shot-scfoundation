## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

def plot_side_by_side(df, label, col_palette=None, direction="row", 
                      title="UMAP projection of the dataset"):
    """
    Plot a scatter plot for each label in the dataset.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the x and y coordinates of the projection and the
        labels. 
    label : str
        Name of the column containing the labels.
    col_palette : dict, optional
        Dictionary containing the colors for each label. The default is None.
    direction : str, optional
        Direction of the subplots. The default is "row".
    title : str, optional
        Title of the plot. The default is "UMAP projection of the dataset".
    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure containing the scatter plots.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # check how many different labels there are
    labels = sorted(df[label].unique().tolist())
    n_labels = len(labels)
    # check if color pallette is provided and if so, if the length is correct
    if col_palette is not None:
        # check if all labels have a color
        for l in labels:
            assert l in col_palette, "Not all labels have a color assigned."
    
    # Create n_labels scatter plots with different color schemes
    if n_labels > 10:
        print("Warning: more than 10 labels! Ignoring direction argument.")
        n_subplots_sqrt = np.sqrt(n_labels)
        n_cols = int(np.ceil(n_subplots_sqrt))
        n_rows = int(np.ceil(n_labels / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                                figsize=(5 * n_cols, 5 * n_rows))

    elif direction == "row":
        fig, axs = plt.subplots(nrows=n_labels, figsize=(5, n_labels * 5))
    elif direction == "col":
        fig, axs = plt.subplots(ncols=n_labels, figsize=(n_labels * 5, 5))
    else:
        raise ValueError("direction must be either row or col.")
    
    # Flatten the axs array to make it easier to iterate over
    axs = axs.flatten()
    
    # loop over the labels
    for i, label_ in enumerate(labels):
        # check if color pallette is provided
        if col_palette is not None:
            label_col = col_palette[label_]
        else:
            label_col = "red"

        # add background scatter
        axs[i].scatter(
            df.loc[df[label] != label_, 'x'], 
            df.loc[df[label] != label_, 'y'], 
            color = "gray", 
            alpha = 0.2,
            s = 0.5
        )

        # add scatter for each label
        axs[i].scatter(
            df.loc[df[label] == label_, 'x'], 
            df.loc[df[label] == label_, 'y'], 
            color = label_col, 
            alpha = 0.5,
            s = 0.5
        )

        # switch off axis
        axs[i].axis('off')

        # add title
        axs[i].title.set_text(label_)
    
    # add main title
    fig.suptitle(title, fontsize=16)

    return fig

def plot_latent(latent, annot_df, label, col_palette=None, direction="row", 
                title="UMAP projection of the dataset"):
    import pandas as pd
    df = pd.DataFrame(latent, columns=["x", "y"])
    df[label] = annot_df[label].tolist()
    psbs = plot_side_by_side(df, label, 
                             col_palette=col_palette, 
                             direction=direction, title=title)
    return psbs

cmaps = {
    'Perceptually Uniform Sequential': [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ],
    'Sequential': [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ],
    'Sequential (2)': [
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper'
    ],
    'Diverging': [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    ],
    'Cyclic': ['twilight', 'twilight_shifted', 'hsv'],
    'Qualitative': [
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c'
    ],
    'Miscellaneous': [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
        'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
        'gist_ncar'
    ]
}

# get random value from a list
def get_random_value_from_list(l):
    """
    Get a random value from a list.
    """
    import random
    return l[random.randint(0, len(l)-1)]

# generate a color pallette of length n
def generate_pallette(n, cmap="viridis"):
    """
    Generate a color pallette of length n.
    """
    import numpy as np
    import seaborn as sns
    if cmap == "random":
        return np.random.rand(n, 3)
    else:
        return sns.color_palette(cmap, n_colors=n).as_hex()

