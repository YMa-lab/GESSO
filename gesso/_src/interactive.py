import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
import pandas as pd
from typing import Literal, Any


class GeneSetActivityScoresReport:
    """Report object for storing GESSO geneset activity score results."""

    def __init__(
        self,
        gas_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        geneset_to_metagene_df_dict: dict,
    ) -> None:
        """Initializes the GeneSetActivityScoresReport object.

        Parameters
        ----------
        gas_df : pd.DataFrame
            geneset activity scores DataFrame. Should be of size (n_obs, n_genesets).

        location_df : pd.DataFrame
            Locations DataFrame. Should be of size (n_obs, 2).

        geneset_to_metagene_df_dict: dict
            Dictionary of geneset to metagene DataFrames.
        """
        self._gas_df = gas_df
        self._location_df = locations_df
        self._orig_spot_order = locations_df.index
        self._geneset_to_metagene_df_dict: dict[str, pd.DataFrame] = (
            geneset_to_metagene_df_dict
        )
        self._n_examples, self._n_genesets = gas_df.shape

    def metagene_df(
        self,
        geneset: str,
        sort_by: Literal["metagene_weight", "gene_name"] = "metagene_weight",
    ) -> pd.DataFrame:
        """Returns a metagene DataFrame with a single column (geneset name).
        The index is the gene name.

        Parameters
        ----------
        geneset : str
            geneset name.

        sort_by : Literal["metagene_weight", "gene_name"]
            Default: "metagene_weight". How to sort the DataFrame.
            If "metagene_weight", sorts by the metagene weight (descending).
            If "gene_name", sorts by the gene name (ascending).

        Returns
        -------
        pd.DataFrame
        """
        output = self._geneset_to_metagene_df_dict[geneset]
        if sort_by == "metagene_weight":
            output = output.sort_values(by=geneset, ascending=False)
        elif sort_by == "gene_name":
            # the gene name is in the index
            output = output.sort_index(ascending=True)
        else:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. "
                "Must be 'metagene_weight' or 'gene_name'."
            )
        return output

    def locations_df(self) -> pd.DataFrame:
        """Returns the locations DataFrame.
        The index is the spot ID. The columns are "x" and "y".

        Returns
        -------
        pd.DataFrame
        """
        return self._location_df[["x", "y"]]

    def gas_df(self) -> pd.DataFrame:
        """Returns the geneset activity scores as a DataFrame.
        The index is the spot ID. The columns are the geneset names.

        Returns
        -------
        pd.DataFrame
        """
        return self._gas_df.loc[self._orig_spot_order]

    def plot_gas_spatial_map(
        self,
        geneset: str,
        size: int = 20,
        cmap: Colormap | str = "viridis",
        show_coords: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: Axes | None = None,
    ) -> Figure:
        """Plots the geneset activity scores of a given geneset of interest
        across all locations.

        Parameters
        ----------
        geneset : str
            The name of the geneset to plot.

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
        plotting_df = self._location_df.join(self._gas_df[geneset])
        cdata = plotting_df[geneset].to_numpy()
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
        ax.set_title(f"GESSO Gene Set Activity Scores")
        fig.tight_layout()
        plt.close(fig)
        return fig


class PermutationTestReport:
    """Report object for storing GESSO permutation test results."""

    def __init__(
        self,
        geneset: str,
        permutation_test_df: pd.DataFrame,
    ):
        """Initializes the PermutationTestReport object.

        Parameters
        ----------
        geneset : str
            The name of the geneset for which the permutation test was performed.

        permuation_test_df : pd.DataFrame
            DataFrame containing the results of the permutation test.
            Should have columns: 'x', 'y', 'gas', 'p'
        """
        self._geneset = geneset
        self._permutation_test_df = permutation_test_df

    def plot_gas_spatial_map(
        self,
        size: int = 20,
        cmap: Colormap | str = "viridis",
        show_coords: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: Axes | None = None,
    ) -> Figure:
        """Plots the gene set activity scores of the permutation
        test across all locations.

        Parameters
        ----------
        size : int
            Default: 20. The size of the scatter points.

        cmap : Colormap | None
            Default: "viridis". The colormap to use for the scatter plot.

        show_coords : bool
            Default: False. If True, shows the coordinates of the points.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). The size of the figure.

        ax : plt.Axes | None
            Default: None. If None, creates a new figure.
        """
        if cmap is None:
            cmap = "viridis"
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        plotting_df = self._permutation_test_df

        cdata = plotting_df[self._geneset].to_numpy()
        scatter = ax.scatter(
            x=plotting_df["x"],
            y=plotting_df["y"],
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
        ax.set_title(f"GESSO: Permutation Test Results for {self._geneset}")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_pval_spatial_map(
        self,
        size: int = 20,
        significance_threshold: float = 0.05,
        significant_color: str | Any = "#800080",  # dark purple hex code
        not_significant_color: str | Any = "#D3D3D3",
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
            Default: "#800080". The color for significant points.

        not_significant_color : str
            Default: "#D3D3D3". The color for not significant points.

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
        ax.scatter(
            x=plotting_df["x"],
            y=plotting_df["y"],
            c=colors,
            s=size,
        )
        ax.set_title(f"GESSO: Spots with Elevated Activity")

        if not show_coords:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout()
        plt.close(fig)
        return fig

    def gas_df(self) -> pd.DataFrame:
        """Returns the geneset activity scores DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the geneset activity scores.
        """
        return self._permutation_test_df[["gas"]].rename(columns={"gas": self._geneset})

    def pval_df(self) -> pd.DataFrame:
        """Returns the p-values DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the p-values.
        """
        return self._permutation_test_df[["p"]]

    def locations_df(self) -> pd.DataFrame:
        """Returns the locations DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the locations of the spots.
            Contains two columns: 'x' and 'y'.
        """
        return self._permutation_test_df[["x", "y"]]

    def htest_df(self) -> pd.DataFrame:
        """Returns the full permutation test DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the full permutation test results.
            Contains four columns: 'x', 'y', 'gas', 'p'.
        """
        return self._permutation_test_df
