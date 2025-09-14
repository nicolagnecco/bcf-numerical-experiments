import datetime
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def split_features(df: pd.DataFrame):
    """
    Splits a DataFrame into continuous and categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        continuous_df (pd.DataFrame): DataFrame containing continuous features.
        categorical_df (pd.DataFrame): DataFrame containing categorical features.
    """

    continuous_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    continuous_df = df[continuous_cols]
    categorical_df = df[categorical_cols]

    return continuous_df, categorical_df


def add_interfix_to_filename(filename, interfix):
    """
    Add an interfix to a filename before its extension.

    Args:
    filename (str): The original filename.
    interfix (str): The string to insert before the file extension.

    Returns:
    str: The new filename with the interfix added.
    """
    # Split the filename by the last dot to separate the extension
    parts = filename.rsplit(".", 1)

    # If there is no extension, return the filename with the interfix appended
    if len(parts) == 1:
        return f"{filename}_{interfix}"

    # Otherwise, insert the interfix before the extension
    name, extension = parts
    new_filename = f"{name}_{interfix}.{extension}"
    return new_filename


def create_timestamped_filename(base_filename):
    """
    Appends a timestamp to a filename.

    Args:
        base_filename (str): The base filename to which the timestamp will be appended.

    Returns:
        str: A new filename with a timestamp appended.
    """
    # Get current date and time as a string (e.g., 20230703-153045)
    timestamp = get_current_timestamp()

    # Create a new filename with the timestamp
    new_filename = add_interfix_to_filename(base_filename, timestamp)

    return new_filename


def create_timestamped_folder(target_root_directory: str, timestamp: str) -> str:
    """
    Creates a new folder named with the given timestamp.

    Args:
        target_root_directory (str): The root directory where the new timestamped folder will be created.
        timestamp             (str): The given timestamp.

    Returns:
        str: The path of the created folder.
    """
    # Create a new directory for this timestamp
    new_directory = os.path.join(target_root_directory, timestamp)
    os.makedirs(new_directory, exist_ok=True)

    return new_directory


def get_current_timestamp() -> str:

    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time as a string (e.g., 20230703-153045)
    return now.strftime("%Y%m%d-%H%M%S")


def plot_data(X, y, f):
    # Define grid
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # Evaluate f on the grid
    Z_surf = f(X_grid).reshape(X1.shape)

    # --- build interactive figure
    fig = go.Figure()

    # surface for f(x)
    fig.add_trace(
        go.Surface(x=X1, y=X2, z=Z_surf, opacity=0.7, showscale=True, name="f(x)")
    )

    # scatter points for generated data
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=y,
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.15),
            name="samples",
        )
    )

    fig.update_layout(
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
        title="Interactive 3D view of f(x) and generated data",
    )

    fig.show()

    fig.show()
