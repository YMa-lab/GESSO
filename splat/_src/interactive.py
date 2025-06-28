import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
import pandas as pd
from typing import Literal


class PathwayActivityScoresReport:
    """Report object for storing SPLAT pathway activity score results."""

    def __init__(
        self,
        pas_df: pd.DataFrame,
        location_df: pd.DataFrame,
        pathway_to_metagene_df_dict: dict,
    ) -> None:
        """Initializes the PathwayActivityScoresReport object.

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
        self._orig_spot_order = location_df.index
        self._pathway_to_metagene_df_dict: dict[str, pd.DataFrame] = (
            pathway_to_metagene_df_dict
        )
        self._n_examples, self._n_pathways = pas_df.shape

    def metagene_df(
        self,
        pathway: str,
        sort_by: Literal["metagene_weight", "gene_name"] = "metagene_weight",
    ) -> pd.DataFrame:
        """Returns a metagene DataFrame with a single column (pathway name).
        The index is the gene name.

        Parameters
        ----------
        pathway : str
            Pathway name.

        sort_by : Literal["metagene_weight", "gene_name"]
            Default: "metagene_weight". How to sort the DataFrame.
            If "metagene_weight", sorts by the metagene weight (descending).
            If "gene_name", sorts by the gene name (ascending).

        Returns
        -------
        pd.DataFrame
        """
        output = self._pathway_to_metagene_df_dict[pathway]
        if sort_by == "metagene_weight":
            output = output.sort_values(by=pathway, ascending=False)
        elif sort_by == "gene_name":
            # the gene name is in the index
            output = output.sort_index(ascending=True)
        else:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. "
                "Must be 'metagene_weight' or 'gene_name'."
            )
        return output

    def location_df(self) -> pd.DataFrame:
        """Returns the location DataFrame.
        The index is the spot ID. The columns are "x" and "y".

        Returns
        -------
        pd.DataFrame
        """
        return self._location_df[["x", "y"]]

    def pas_df(self) -> pd.DataFrame:
        """Returns the pathway activity scores as a DataFrame.
        The index is the spot ID. The columns are the pathway names.

        Returns
        -------
        pd.DataFrame
        """
        return self._pas_df.loc[self._orig_spot_order]

    def plot_pas_spatial_map(
        self,
        pathway: str,
        size: int = 20,
        cmap: Colormap | str = "viridis",
        show_coords: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: Axes | None = None,
    ) -> Figure:
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

        show_coords : bool
            Default: False. If True, shows the coordinates of the points.

        figsize : tuple[float, float]
            Default: (5.0, 5.0) The size of the figure.

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
            vmin=cdata.min(),
            vmax=cdata.max(),
        )
        if not show_coords:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.01)
        ax.set_title(f"SPLAT Pathway Activity Scores")
        fig.tight_layout()
        plt.close(fig)
        return fig


class PermutationTestReport:
    """Report object for storing SPLAT permutation test results."""

    def __init__(
        self,
        pathway: str,
        permuation_test_df: pd.DataFrame,
    ):
        """Initializes the PermutationTestReport object.

        Parameters
        ----------
        pathway : str
            The name of the pathway for which the permutation test was performed.

        permuation_test_df : pd.DataFrame
            DataFrame containing the results of the permutation test.
            Should have columns: 'x', 'y', 'pas', 'p'
        """
        self._pathway = pathway
        self._permutation_test_df = permuation_test_df

    def plot_pval_spatial_map(
        self,
        size: int = 20,
        significance_threshold: float = 0.05,
        significant_color: str = "purple",
        not_significant_color: str = "gray",
        show_coords: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: Axes | None = None,
    ) -> Figure:
        """Plots the p-values of the permutation test across all locations.

        Parameters
        ----------
        size : int
            Default: 20. The size of the scatter points.

        significance_threshold : float
            Default: 0.05. The threshold for significance.

        significant_color : str
            Default: "purple". The color for significant points.

        not_significant_color : str
            Default: "gray". The color for not significant points.

        show_coords : bool
            Default: False. If True, shows the coordinates of the points.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). The size of the figure.

        ax : plt.Axes | None
            Default: None. If None, creates a new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        plotting_df = self._permutation_test_df

        colors = [
            significant_color if p < significance_threshold else not_significant_color
            for p in plotting_df["p"]
        ]
        scatter = ax.scatter(
            x=plotting_df["x"],
            y=plotting_df["y"],
            c=colors,
            s=size,
        )
        ax.set_title(f"SPLAT: Spots with Elevated Activity")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        plt.close(fig)
        return fig
