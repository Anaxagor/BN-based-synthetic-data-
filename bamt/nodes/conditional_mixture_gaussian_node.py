import itertools
from typing import Union, List, Optional, Dict

import numpy as np
from gmr import GMM
from pandas import DataFrame

from bamt.utils.MathUtils import component
from .base import BaseNode
from .schema import CondMixtureGaussParams


class ConditionalMixtureGaussianNode(BaseNode):
    """
    Main class for Conditional Mixture Gaussian Node
    """

    def __init__(self, name):
        super(ConditionalMixtureGaussianNode, self).__init__(name)
        self.type = "ConditionalMixtureGaussian"

    def fit_parameters(
        self, data: DataFrame
    ) -> Dict[str, Dict[str, CondMixtureGaussParams]]:
        """
        Train params for Conditional Mixture Gaussian Node.
        Return:
        {"hybcprob": {<combination of outputs from discrete parents> : CondMixtureGaussParams}}
        """
        hycprob = dict()
        values = []
        combinations = []
        for d_p in self.disc_parents:
            values.append(np.unique(data[d_p].values))
        for xs in itertools.product(*values):
            combinations.append(list(xs))
        for comb in combinations:
            mask = np.full(len(data), True)
            for col, val in zip(self.disc_parents, comb):
                mask = (mask) & (data[col] == val)
            new_data = data[mask]
            new_data.reset_index(inplace=True, drop=True)
            key_comb = [str(x) for x in comb]
            nodes = [self.name] + self.cont_parents
            if new_data.shape[0] > 5:
                if self.cont_parents:
                    # component(new_data, nodes,
                    # 'LRTS')#int((component(new_data, nodes, 'aic') +
                    # component(new_data, nodes, 'bic')) / 2)
                    n_comp = int(
                        (
                            component(new_data, nodes, "aic")
                            + component(new_data, nodes, "bic")
                        )
                        / 2
                    )
                    # n_comp = 3
                    gmm = GMM(n_components=n_comp).from_samples(
                        new_data[nodes].values, n_iter=500, init_params="kmeans++"
                    )
                else:
                    # component(new_data, [node],
                    # 'LRTS')#int((component(new_data, [node], 'aic') +
                    # component(new_data, [node], 'bic')) / 2)
                    n_comp = int(
                        (
                            component(new_data, [self.name], "aic")
                            + component(new_data, [self.name], "bic")
                        )
                        / 2
                    )
                    # n_comp = 3
                    gmm = GMM(n_components=n_comp).from_samples(
                        np.transpose([new_data[self.name].values]),
                        n_iter=500,
                        init_params="kmeans++",
                    )
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(np.transpose([new_data[node].values])))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                hycprob[str(key_comb)] = {"covars": cov, "mean": means, "coef": w}
            elif new_data.shape[0] != 0:
                n_comp = 1
                gmm = GMM(n_components=n_comp)
                if self.cont_parents:
                    gmm.from_samples(new_data[nodes].values)
                else:
                    gmm.from_samples(np.transpose([new_data[self.name].values]))
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(np.transpose([new_data[node].values])))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                hycprob[str(key_comb)] = {"covars": cov, "mean": means, "coef": w}
            else:
                if self.cont_parents:
                    hycprob[str(key_comb)] = {
                        "covars": np.nan,
                        "mean": np.nan,
                        "coef": [],
                    }
                else:
                    hycprob[str(key_comb)] = {
                        "covars": np.nan,
                        "mean": np.nan,
                        "coef": [],
                    }
        return {"hybcprob": hycprob}

    @staticmethod
    def get_dist(node_info, pvals):
        lgpvals = []
        dispvals = []

        for pval in pvals:
            if (isinstance(pval, str)) | (isinstance(pval, int)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)

        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        covariance = lgdistribution["covars"]
        w = lgdistribution["coef"]

        if len(w) != 0:
            if len(lgpvals) != 0:
                indexes = [i for i in range(1, (len(lgpvals) + 1), 1)]
                if not np.isnan(np.array(lgpvals)).all():
                    n_comp = len(w)
                    gmm = GMM(
                        n_components=n_comp,
                        priors=w,
                        means=mean,
                        covariances=covariance,
                    )
                    cond_gmm = gmm.condition(indexes, [lgpvals])
                    return cond_gmm.means, cond_gmm.covariances, cond_gmm.priors
                else:
                    return np.nan, np.nan, np.nan
            else:
                n_comp = len(w)
                gmm = GMM(
                    n_components=n_comp, priors=w, means=mean, covariances=covariance
                )
                return gmm.means, gmm.covariances, gmm.priors
        else:
            return np.nan, np.nan, np.nan

    def choose(
        self,
        node_info: Dict[str, Dict[str, CondMixtureGaussParams]],
        pvals: List[Union[str, float]],
    ) -> Optional[float]:
        """
        Function to get value from ConditionalMixtureGaussian node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        mean, covariance, w = self.get_dist(node_info, pvals)

        # check if w is nan or list of weights
        if not isinstance(w, np.ndarray):
            return np.nan

        n_comp = len(w)

        gmm = GMM(
            n_components=n_comp,
            priors=w,
            means=mean,
            covariances=covariance,
        )
        sample = gmm.sample(1)[0][0]
        return sample

    @staticmethod
    def predict(
        node_info: Dict[str, Dict[str, CondMixtureGaussParams]],
        pvals: List[Union[str, float]],
    ) -> Optional[float]:
        """
        Function to get prediction from ConditionalMixtureGaussian node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        dispvals = []
        lgpvals = []
        for pval in pvals:
            if (isinstance(pval, str)) | (isinstance(pval, int)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        covariance = lgdistribution["covars"]
        w = lgdistribution["coef"]
        if len(w) != 0:
            if len(lgpvals) != 0:
                indexes = [i for i in range(1, (len(lgpvals) + 1), 1)]
                if not np.isnan(np.array(lgpvals)).all():
                    n_comp = len(w)
                    gmm = GMM(
                        n_components=n_comp,
                        priors=w,
                        means=mean,
                        covariances=covariance,
                    )
                    sample = gmm.predict(indexes, [lgpvals])[0][0]
                else:
                    sample = np.nan
            else:
                # n_comp = len(w)
                # gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                sample = 0
                for ind, wi in enumerate(w):
                    sample += wi * mean[ind][0]
        else:
            sample = np.nan
        return sample
