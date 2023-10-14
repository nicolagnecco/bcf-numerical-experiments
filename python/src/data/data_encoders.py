import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def prepare_Z(Z) -> np.ndarray:
    """
    Encodes categorical columns of Z using OneHotEncoder if they are not numeric.

    Parameters:
    ----------
    Z : pandas.DataFrame
        DataFrame containing the instrumental/exogenous variables, which may be categorical.

    Returns:
    --------
    numpy.ndarray
        An array of transformed Z values ready for use in modeling.
    """
    # Check if Z is a DataFrame and convert if necessary
    if isinstance(Z, pd.DataFrame):
        categorical_features = Z.select_dtypes(include=["object", "category"]).columns
        if not categorical_features.empty:
            encoder = OneHotEncoder(drop="first", sparse=False)
            # Fit and transform the categorical columns
            Z_encoded = encoder.fit_transform(Z[categorical_features])
            # Drop original categorical columns and concatenate the encoded ones
            Z_ = np.hstack((Z.drop(columns=categorical_features).values, Z_encoded))
    elif isinstance(Z, np.ndarray):
        # Assuming Z is already prepared if it's an ndarray
        Z_ = Z
    else:
        raise ValueError("Z must be a pandas DataFrame or a numpy ndarray.")

    return Z_
