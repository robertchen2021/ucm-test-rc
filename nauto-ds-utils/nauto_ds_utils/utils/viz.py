
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional


def plot_scatter(df: pd.DataFrame,
                 x: str,
                 y: str,
                 z: str,
                 figsize=(13, 10),
                 alpha: float = 1/2,
                 fontsize: int = 13,
                 title: Optional[str] = None,
                 title_fontsize: int = 16) -> None:
    """Scatter plot for features in a Pandas dataframe. This requires three
    attributes.
    """
    plt.figure(figsize=figsize)
    plt.scatter(df[x], df[y], c=df[z], alpha=alpha)
    plt.xlabel(x, fontsize=fontsize)
    plt.ylabel(y, fontsize=fontsize)
    plt.colorbar().set_label(z, fontsize=fontsize)
    if title is None:
        title = f'{x} vs {y}'
    plt.title(title, fontsize=title_fontsize)
    return
