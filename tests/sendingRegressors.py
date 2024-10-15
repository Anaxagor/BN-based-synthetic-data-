# import json

import pandas as pd
from catboost import CatBoostRegressor
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

import bamt.preprocessors as preprocessors
from bamt.networks import HybridBN

hack_data = pd.read_csv("../data/real data/hack_processed_with_rf.csv")[
    [
        "Tectonic regime",
        "Period",
        "Lithology",
        "Structural setting",
        "Gross",
        "Netpay",
        "Porosity",
        "Permeability",
        "Depth",
    ]
].dropna()

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")

p = preprocessors.Preprocessor([("encoder", encoder), ("discretizer", discretizer)])

discretized_data, est = p.apply(hack_data)

bn = HybridBN(has_logit=True)
info = p.info

# with open(r"C:\Users\Roman\Desktop\mymodels\mynet.json") as f:
#     net_data = json.load(f)

# bn.add_nodes(net_data["info"])
# bn.set_structure(edges=net_data["edges"])
# bn.set_parameters(net_data["parameters"])

bn.add_nodes(info)

bn.add_edges(discretized_data, scoring_function=("BIC",), progress_bar=False)

bn.set_regressor(
    regressors={
        "Depth": CatBoostRegressor(logging_level="Silent", allow_writing_files=False),
        "Gross": RandomForestRegressor(),
        "Porosity": DecisionTreeRegressor(),
    }
)

bn.fit_parameters(hack_data)

# bn.save("mynet")

print(bn.sample(100).shape)
bn.get_info(as_df=False)
