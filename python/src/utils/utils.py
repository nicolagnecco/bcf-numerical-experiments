import numpy as np
import pandas as pd


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
