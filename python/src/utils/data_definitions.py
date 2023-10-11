import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Dataset:
    """Representation of a dataset

    Attributes
    ----------
    X : np.ndarray
        feature matrix with n rows and d columns

    y : np.ndarray
        response vector with n observations

    Z : np.ndarray
        instrument matrix with n rows and K columns

    V : np.ndarray or None
        control variable matrix with n rows and d columns

    Assumptions
    -----------
    X, y, Z, (V) have same number of rows.
    """

    X: np.ndarray
    y: np.ndarray
    Z: np.ndarray
    V: Optional[np.ndarray] = None

    @property
    def nrows(self) -> int:
        """
        Dataset -> integer
        Returns number of rows in `X` (which is the same as `y` or `Z`)
        """
        return self.X.shape[0]


@dataclass
class Tree:
    """Representation of a decision tree

    Attributes
    ----------
    feature : int, default=None
        holds the feature at which the split occurs

    threshold : float, default=None
        holds the value at which the split occurs

    impurity : float, default=None
        holds the impurity of the node

    value : float, default=None
        holds the constant prediction value of the node

    n_node_sample : int, default=None
        holds the number of observations in the sample

    dataset : Dataset, default=None
        holds the data of the node if it is terminal

    left : Tree, default=None
        holds the left sub-tree

    right : Tree, default=None
        holds the right sub-tree
    """

    feature: Optional[int] = None
    threshold: Optional[float] = None
    impurity: Optional[float] = None
    value: Optional[float] = None
    n_node_samples: Optional[int] = None
    dataset: Optional[Dataset] = None
    left: Optional["Tree"] = None
    right: Optional["Tree"] = None

    def add_node(self, *args):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        np.ndarray(n, d) -> np.ndarray(n,)
        Predicts values at X given a fitted tree
        """
        return np.apply_along_axis(self.predict_helper, 1, X)

    def predict_helper(self, x: np.ndarray) -> np.ndarray:
        """
        np.ndarray(1, d) -> np.ndarray(1,)
        """
        if self.is_terminal:
            return np.array(self.value)
        elif self.left is None or self.right is None:
            raise Exception(
                "There is a node with only one child. This is not allowed in binary trees."
            )
        else:
            if x[self.feature] <= self.threshold:
                return self.left.predict_helper(x)
            else:
                return self.right.predict_helper(x)

    @property
    def is_terminal(self) -> bool:
        return self.left is None and self.right is None

    def __repr__(self) -> str:
        return self.print()

    def print(self, depth: int = 0) -> str:

        if self == Tree():
            return "Tree()"

        if self.is_terminal:
            return "{}Y:{}, N:{}\n".format(
                depth * ".", np.round(self.value, 3), self.n_node_samples
            )
        elif self.left is None or self.right is None:
            raise Exception(
                "There is a node with only one child. This is not allowed in binary trees."
            )
        else:
            curr_str = "{}X_{}: {}\n".format(
                depth * ".", self.feature, np.round(self.threshold, 3)
            )
            left_str = self.left.print(depth=depth + 1)
            right_str = self.right.print(depth=depth + 1)
            return curr_str + left_str + right_str

    def prune(self):
        pass


@dataclass
class Split:
    """Representation of a split

    Attributes
    ----------
    feature : int, default=None
        holds the feature at which the split occurs

    threshold : float, default=None
        holds the value at which the split occurs

    data_left : Dataset, default=None
        holds the dataset of the left node

    data_right : Dataset, default=None
        holds the dataset of the right node

    impurity_left : float, default=np.Inf
        holds the impurity of the left node

    impurity_right : float, default=np.Inf
        holds the impurity of the right node

    impurity_decrease : float, default=np.Inf
        holds the weighted impurity decrease of the split, given by

             (N_t * impurity - N_t_R * impurity_left
                        - N_t_L * impurity_right),

        where ``N_t_L`` is the number of samples in `data_left`,
        ``N_t_R`` is the number of samples in the `data_right`,
        ``N_t = N_t_L + N_t_R``, and ``impurity`` is the impurity of the parent node

    value_left: float, default=None
        holds the fitted value of the left node

    value_right: float, default=None
        holds the fitted value of the right node

    has_failed: bool, default=False
        whether the split was successful or not

    """

    feature: Optional[int] = None
    threshold: Optional[float] = None
    data_left: Optional[Dataset] = None
    data_right: Optional[Dataset] = None
    impurity_left: float = np.Inf
    impurity_right: float = np.Inf
    impurity_decrease: float = np.Inf
    value_left: Optional[float] = None
    value_right: Optional[float] = None
    has_failed: bool = False
