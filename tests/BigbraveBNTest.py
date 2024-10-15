import pandas as pd
from pgmpy.estimators import K2Score
from sklearn import preprocessing

import bamt.preprocessors as pp
from bamt.networks.big_brave_bn import BigBraveBN
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.discrete_bn import DiscreteBN

data_discrete = pd.read_csv('https://raw.githubusercontent.com/jrzkaminski/BAMT-old/main/data/pigs.csv')
data_continuous = pd.read_csv('https://raw.githubusercontent.com/jrzkaminski/BAMT-old/main/data/arth150.csv')

encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(
    n_bins=5, encode="ordinal", strategy="uniform"
)

p = pp.Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
discretized_data, est = p.apply(data_discrete)

info = p.info

space_restrictor = BigBraveBN()

space_restrictor.set_possible_edges_by_brave(df=data_discrete)

ps = space_restrictor.possible_edges

bn_discrete = DiscreteBN()

bn_discrete.add_nodes(descriptor=info)

params = {"white_list": ps}
bn_discrete.add_edges(discretized_data, scoring_function=("K2", K2Score), params=params)

encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(
    n_bins=5, encode="ordinal", strategy="uniform"
)

p = pp.Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
discretized_data, est = p.apply(data_continuous)

info = p.info

space_restrictor = BigBraveBN()

space_restrictor.set_possible_edges_by_brave(df=data_continuous)

ps = space_restrictor.possible_edges

bn_continuous = ContinuousBN()

bn_continuous.add_nodes(descriptor=info)

params = {"white_list": ps}
bn_continuous.add_edges(
    discretized_data, scoring_function=("K2", K2Score), params=params
)
