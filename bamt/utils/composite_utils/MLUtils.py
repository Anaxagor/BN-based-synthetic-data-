import json
from random import choice
from typing import Union

import pkg_resources
from catboost import CatBoostClassifier, CatBoostRegressor
from golem.core.dag.graph_node import GraphNode
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from .CompositeModel import CompositeNode

# Try to import LGBMRegressor and LGBMClassifier from lightgbm, if not available set to None
try:
    from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
except ModuleNotFoundError:
    LGBMRegressor = None
    LGBMClassifier = None

lgbm_params = "lgbm_params.json"
models_repo = "models_repo.json"
lgbm_params_path = pkg_resources.resource_filename(__name__, lgbm_params)
models_repo_path = pkg_resources.resource_filename(__name__, models_repo)


class MlModels:
    def __init__(self):
        self.operations_by_types = {
            "xgbreg": "XGBRegressor",
            "adareg": "AdaBoostRegressor",
            "gbr": "GradientBoostingRegressor",
            "dtreg": "DecisionTreeRegressor",
            "treg": "ExtraTreesRegressor",
            "rfr": "RandomForestRegressor",
            "linear": "LinearRegression",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "sgdr": "SGDRegressor",
            "lgbmreg": "LGBMRegressor",
            "catboostreg": "CatBoostRegressor",
            "xgboost": "XGBClassifier",
            "logit": "LogisticRegression",
            "bernb": "BernoulliNB",
            "multinb": "MultinomialNB",
            "dt": "DecisionTreeClassifier",
            "rf": "RandomForestClassifier",
            "mlp": "MLPClassifier",
            "catboost": "CatBoostClassifier",
            "kmeans": "KMeans",
        }

        self.dict_models = {
            "XGBRegressor": XGBRegressor,
            "AdaBoostRegressor": AdaBoostRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "SGDRegressor": SGDRegressor,
            "LGBMRegressor": LGBMRegressor,
            "CatBoostRegressor": CatBoostRegressor,
            "XGBClassifier": XGBClassifier,
            "LogisticRegression": LogisticRegression,
            "BernoulliNB": BernoulliNB,
            "MultinomialNB": MultinomialNB,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "MLPClassifier": MLPClassifier,
            "LGBMClassifier": LGBMClassifier,
            "CatBoostClassifier": CatBoostClassifier,
            "KMeans": KMeans,
        }

        # Include LGBMRegressor and LGBMClassifier if they were imported successfully
        if LGBMRegressor is not None:
            self.dict_models["LGBMRegressor"] = LGBMRegressor
            self.operations_by_types["lgbmreg"] = "LGBMRegressor"
        if LGBMClassifier is not None:
            self.dict_models["LGBMClassifier"] = LGBMClassifier
            self.operations_by_types["lgbm"] = "LGBMClassifier"

        if LGBMClassifier and LGBMRegressor is not None:
            with open(lgbm_params_path) as file:
                self.lgbm_dict = json.load(file)

    def get_model_by_children_type(self, node: Union[GraphNode, CompositeNode]):
        candidates = []
        if node.content["type"] == "cont":
            type_model = "regr"
        else:
            type_model = "class"
        forbidden_tags = ["non-default", "expensive"]
        with open(models_repo_path, "r") as f:
            models_json = json.load(f)
            models = models_json["operations"]
            if LGBMClassifier and LGBMRegressor is not None:
                models = models | self.lgbm_dict
            for model, value in models.items():
                if (
                    model not in ["knnreg", "knn", "qda"]
                    and list(set(value["tags"]).intersection(forbidden_tags)) == []
                    and type_model in value["meta"]
                ):
                    candidates.append(model)
        return self.operations_by_types[choice(candidates)]
