import random
from itertools import product
from typing import Type, Dict, Union, List

import numpy as np
from pandas import DataFrame, crosstab

from .base import BaseNode
from .schema import DiscreteParams


class DiscreteNode(BaseNode):
    """
    Main class of Discrete Node
    """

    def __init__(self, name):
        super(DiscreteNode, self).__init__(name)
        self.type = "Discrete"

    def fit_parameters(self, data: DataFrame, num_workers: int = 1):
        """
        Train params for Discrete Node
        data: DataFrame to train on
        num_workers: number of Parallel Workers
        Method returns probas dict with the following format {[<combinations>: value]}
        and vals, list of appeared values in combinations
        """

        def worker(node: Type[BaseNode]) -> DiscreteParams:
            parents = node.disc_parents + node.cont_parents
            dist = data[node.name].value_counts(normalize=True).sort_index()
            vals = [str(i) for i in dist.index.to_list()]

            if not parents:
                cprob = dist.to_list()
            else:
                cprob = {
                    str([str(i) for i in comb]): [1 / len(vals) for _ in vals]
                    for comb in product(*[data[p].unique() for p in parents])
                }

                conditional_dist = crosstab(
                    data[node.name].to_list(),
                    [data[p] for p in parents],
                    normalize="columns",
                ).T
                tight_form = conditional_dist.to_dict("tight")

                for comb, probs in zip(tight_form["index"], tight_form["data"]):
                    if len(parents) > 1:
                        cprob[str([str(i) for i in comb])] = probs
                    else:
                        cprob[f"['{comb}']"] = probs
            return {"cprob": cprob, "vals": vals}

        # pool = ThreadPoolExecutor(num_workers)
        # future = pool.submit(worker, self)
        result = worker(self)
        return result

    @staticmethod
    def get_dist(node_info, pvals):
        if not pvals:
            return node_info["cprob"]
        else:
            # noinspection PyTypeChecker
            return node_info["cprob"][str(pvals)]

    def choose(self, node_info: Dict[str, Union[float, str]], pvals: List[str]) -> str:
        """
        Return value from discrete node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        vals = node_info["vals"]
        dist = np.array(self.get_dist(node_info, pvals))

        cumulative_dist = np.cumsum(dist)

        rand = np.random.random()
        rindex = np.searchsorted(cumulative_dist, rand)

        return vals[rindex]

    @staticmethod
    def searchsorted_per_row(row, rand_num):
        return np.searchsorted(row, rand_num, side="right")

    def vectorized_choose(
        self,
        node_info: Dict[str, Union[float, str]],
        pvals_array: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Vectorized method to return values from a discrete node.
        params:
        node_info: node's info from distributions
        pvals_array: array of parent values, each row corresponds to a set of parent values
        n_samples: number of samples to generate
        """
        vals = node_info["vals"]

        # Generate a matrix of distributions
        if pvals_array is None or len(pvals_array) == 0:
            # Handle the case with no parent nodes
            dist = np.array(node_info["cprob"])
            dist_matrix = np.tile(dist, (n_samples, 1))
        else:
            # Ensure pvals_array is limited to current batch size
            pvals_array = pvals_array[:n_samples]
            # Compute distribution for each set of parent values
            dist_matrix = np.array(
                [self.get_dist(node_info, pvals.tolist()) for pvals in pvals_array]
            )

        # Ensure that dist_matrix is 2D
        if dist_matrix.ndim == 1:
            dist_matrix = dist_matrix.reshape(1, -1)

        # Generate cumulative distributions
        cumulative_dist_matrix = np.cumsum(dist_matrix, axis=1)

        random_nums = np.random.rand(n_samples)

        # Apply searchsorted across each row
        indices = np.apply_along_axis(
            self.searchsorted_per_row, 1, cumulative_dist_matrix, random_nums
        )

        if indices.ndim > 1:
            indices = indices.flatten()

        sampled_values = np.array(vals)[indices]

        return sampled_values

    @staticmethod
    def predict(node_info: Dict[str, Union[float, str]], pvals: List[str]) -> str:
        """function for prediction based on evidence values in discrete node

        Args:
            node_info (Dict[str, Union[float, str]]): parameters of node
            pvals (List[str]): values in parents nodes

        Returns:
            str: prediction
        """

        vals = node_info["vals"]
        disct = []
        if not pvals:
            dist = node_info["cprob"]
        else:
            # noinspection PyTypeChecker
            dist = node_info["cprob"][str(pvals)]
        max_value = max(dist)
        indices = [index for index, value in enumerate(dist) if value == max_value]
        max_ind = 0
        if len(indices) == 1:
            max_ind = indices[0]
        else:
            max_ind = random.choice(indices)
        return vals[max_ind]
