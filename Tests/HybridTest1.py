import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../Data/hack_processed_with_rf.csv")

cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
h = h[cols]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(h)
info = p.info

print("has_logit=False, use_mixture=False")
bn = Networks.HybridBN()
bn.add_nodes(descriptor=info)

for node in bn.nodes:
    print(f"{node.name}: {node.type}") # only gaussian and discrete nodes
print("#"*1000)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)

for node in bn.nodes:
    print(f"{node.name}: {node.type}. Disc_parents: {len(node.disc_parents)}, cont_parents: {len(node.cont_parents)}") # only gaussian and discrete nodes
print("#"*1000)

print("has_logit=True, use_mixture=False")
bn = Networks.HybridBN(has_logit=True)
bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)
for node in bn.nodes:
    print(f"{node.name}: {node.type}. Disc_parents: {len(node.disc_parents)}, cont_parents: {len(node.cont_parents)}")

print("#"*1000)
print("has_logit=True, use_mixture=True")
bn = Networks.HybridBN(has_logit=True, use_mixture=True)
bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)
for node in bn.nodes:
    print(f"{node.name}: {node.type}. Disc_parents: {len(node.disc_parents)}, cont_parents: {len(node.cont_parents)}")

# t1 = time.time()
# bn.fit_parameters(data=h)
# t2 = time.time()
# print(f'PL elaspsed: {t2-t1}')
# for node, d in bn.distributions.items():
#     print(node,":", d)
#     break