import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import pandas as pd


class SplatReport:
    """Report object for storing SPLAT results."""

    def __init__(
        self,
        pas_df: pd.DataFrame,
        location_df: pd.DataFrame,
        pathway_to_metagene_df_dict: dict,
    ):
        """Initializes the SplatReport object.

        Parameters
        ----------
        pas_df : pd.DataFrame
            Pathway activity scores DataFrame. Should be of size (n_obs, n_pathways).

        location_df : pd.DataFrame
            Locations DataFrame. Should be of size (n_obs, 2).

        pathway_to_metagene_df_dict: dict
            Dictionary of pathway to metagene DataFrames.
        """
        self._pas_df = pas_df
        self._location_df = location_df
        self._pathway_to_metagene_df_dict = pathway_to_metagene_df_dict
        self._n_examples, self._n_pathways = pas_df.shape

    def metagene_df(self, pathway: str):
        """Returns a metagene DataFrame.

        Parameters
        ----------
        pathway : str
            Pathway name.

        Returns
        -------
        pd.DataFrame
        """
        return self._pathway_to_metagene_df_dict[pathway]

    def location_df(self):
        """Returns the location DataFrame which
        contains 'x' and 'y' columns. The index matches
        that of the activity scores DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self._location_df[["x", "y"]]

    def activity_scores_df(self):
        """Returns the pathway activity scores.

        Returns
        -------
        pd.DataFrame
        """
        return self._pas_df

    def plot_activity_scores(
        self,
        pathway: str,
        size: int = 20,
        cmap: Colormap | str = "viridis",
        figsize: tuple[float, float] = (6.0, 6.0),
        ax: plt.Axes | None = None,
    ):
        """Plots the pathway activity scores of a given pathway of interest
        across all locations.

        Parameters
        ----------
        pathway : str
            The name of the pathway to plot.

        size : int
            Default: 20. The size of the scatter points.

        cmap : Colormap | None
            Default:  "viridis". The colormap to use for the scatter plot.

        figsize : tuple[float, float]
            Default: (6.0, 6.0) The size of the figure.

        ax : plt.Axes | None
            Default: None. If None, creates a new figure.
        """
        if cmap is None:
            cmap = "viridis"
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        plotting_df = self._location_df.join(self._pas_df[pathway])
        cdata = plotting_df[pathway].to_numpy()
        scatter = ax.scatter(
            x=plotting_df["x"].to_numpy(),
            y=plotting_df["y"].to_numpy(),
            c=cdata,
            s=size,
            cmap=cmap,
            vmin=cdata.max(),
            vmax=cdata.min(),
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(rotation=270, labelpad=15)
        ax.set_title(f"PAS for {pathway[:30]}")
        fig.tight_layout()
        plt.close(fig)
        return fig
