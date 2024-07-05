from dataclasses import dataclass
from typing import Any, Optional, Type

import numpy as np
from src.algorithms.oracle_methods import CausalFunction, ConstantFunc, IMPFunction
from src.bcf.boosted_control_function_2 import BCF, OLS


@dataclass
class ModelWrapper:
    model: Any

    def fit(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """
        Fit the model with appropriate data based on its type.

        Parameters:
        -----------
        X : numpy.ndarray
            Predictors matrix.
        Y : numpy.ndarray
            Response variable.
        Z : numpy.ndarray
            Instrumental/exogenous variables.
        """
        # Adjust BCF parameters if the model is BCF
        if isinstance(self.model, BCF):
            self.model.n_exog = Z.shape[1]
            self.model.continuous_mask = np.repeat(True, X.shape[1])

        # Fit model
        if isinstance(self.model, (BCF)):
            # These models explicitly require X and Z concatenated
            self.model.fit(np.hstack([X, Z]), Y)
        elif isinstance(self.model, (IMPFunction, CausalFunction, ConstantFunc)):
            # These models explictly require separate X and Z
            self.model.fit(X, Y, Z)
        elif isinstance(self.model, (OLS)):
            # These models only require X and Y
            self.model.fit(X, Y)
        else:
            raise ValueError("Model type not supported")

    def predict(self, X: np.ndarray, Z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using the model with appropriate data based on its type.

        Parameters:
        -----------
        X : numpy.ndarray
            Predictors matrix.
        Z : numpy.ndarray, optional
            Instrumental/exogenous variables, if needed by the model.

        Returns:
        --------
        numpy.ndarray
            Predicted values.
        """
        if isinstance(self.model, BCF) and (Z is not None):
            # BCF might require a different method or include Z in prediction
            return self.model.predict(np.hstack([X, Z]))
        else:
            # For all other cases
            return self.model.predict(X)
