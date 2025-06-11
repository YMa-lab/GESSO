import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Colormap
import pandas as pd
from pathlib import Path


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

    def save_activity_scores_csv(self, path: Path | str):
        """Saves pathway activity scores DataFrame as a CSV.

        Parameters
        ----------
        path : Path | str
            Pathlike string or Path object.
        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(
                    f"Could not convert path to Path object. Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        if not path.exists():
            try:
                path.mkdir(parents=True)
            except Exception as e:
                raise ValueError(
                    f"Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        self._pas_df.to_csv(path)

    def save_locations_csv(self, path: Path | str):
        """Saves location DataFrame as a CSV.

        Parameters
        ----------
        path : Path | str
            Pathlike string or Path object.
        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(
                    f"Could not convert path to Path object. Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        if not path.exists():
            try:
                path.mkdir(parents=True)
            except Exception as e:
                raise ValueError(
                    f"Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        self._location_df.to_csv(path)

    def save_activity_scores_pkl(self, path: Path | str):
        """Saves pathway activity scores DataFrame as a pickle.

        Parameters
        ----------
        path : Path | str
            Pathlike string or Path object.
        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(
                    f"Could not convert path to Path object. Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        if not path.exists():
            try:
                path.mkdir(parents=True)
            except Exception as e:
                raise ValueError(
                    f"Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        self._pas_df.to_pickle(path)

    def save_locations_pkl(self, path: Path | str):
        """Saves location DataFrame as a pickle.

        Parameters
        ----------
        path : Path | str
            Pathlike string or Path object.
        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(
                    f"Could not convert path to Path object. Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        if not path.exists():
            try:
                path.mkdir(parents=True)
            except Exception as e:
                raise ValueError(
                    f"Error: {str(e)}. "
                    "Please double check the validity of the provided path."
                )
        self._location_df.to_pickle(path)

    def plot_activity_scores(
        self,
        pathway: str,
        size: int = 20,
        cmap: Colormap | None = None,
        figsize: tuple[float, float] = (6.0, 5.0),
        ax: plt.Axes | None = None,
    ):
        """Plots the pathway activity scores of a given pathway of interest
        across all locations.

        Parameters
        ----------
        pathway : str

        size : int
            Default: 20. The size of the scatter points.

        cmap : Colormap | None
            Default: None. If None, uses a default Colormap.

        figsize : tuple[float, float]
            Default: (6.0, 5.0) The size of the figure.

        ax : plt.Axes | None
            Default: None. If None, creates a new figure.
        """
        if cmap is None:
            cmap = ListedColormap(
                ["#8ECAE6", "#219EBC", "#023047", "#FFB703", "#FB8500"]
            )
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
        plt.colorbar(scatter)
        ax.set_title(f"Activity Scores for {pathway}")
        plt.tight_layout()
        plt.close(fig)

        return fig
