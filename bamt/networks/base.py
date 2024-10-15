import json
import os.path as path
import random
import re
from copy import deepcopy
from typing import Dict, Tuple, List, Callable, Optional, Type, Union, Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pgmpy.estimators import K2Score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import bamt.builders as builders
from bamt.builders.builders_base import ParamDict
from bamt.builders.evo_builder import EvoStructureBuilder
from bamt.builders.hc_builder import HCStructureBuilder
from bamt.display import plot_, get_info_
from bamt.external.pyitlib.DiscreteRandomVariableUtils import (
    entropy,
    information_mutual,
    information_mutual_conditional,
    entropy_conditional,
)
from bamt.log import logger_network
from bamt.nodes.base import BaseNode
from bamt.utils import GraphUtils, serialization_utils, check_utils


class BaseNetwork(object):
    """
    Base class for Bayesian Network
    """

    def __init__(self):
        """
        nodes: a list of nodes instances
        edges: a list of edges
        distributions: dict
        """
        self.sf_name = None
        self.type = "Abstract"
        self._allowed_dtypes = ["Abstract"]
        self.nodes = []
        self.edges = []
        self.weights = {}
        self.descriptor = {"types": {}, "signs": {}}
        self.distributions = {}
        self.has_logit = False
        self.use_mixture = False
        self.encoders = {}

    @property
    def nodes_names(self) -> List[str]:
        return [node.name for node in self.nodes]

    def __getitem__(self, node_name: str) -> Type[BaseNode]:
        if node_name in self.nodes_names:
            index = self.nodes_names.index(node_name)
            return self.nodes[index]

    def validate(self, descriptor: Dict[str, Dict[str, str]]) -> bool:
        types = descriptor["types"]
        return (
            True if all([a in self._allowed_dtypes for a in types.values()]) else False
        )

    def update_descriptor(self):
        new_nodes_names = [node.name for node in self.nodes]
        self.descriptor["types"] = {
            node: type
            for node, type in self.descriptor["types"].items()
            if node in new_nodes_names
        }
        if "cont" in self.descriptor["types"].values():
            self.descriptor["signs"] = {
                node: sign
                for node, sign in self.descriptor["signs"].items()
                if node in new_nodes_names
            }

    def add_nodes(self, descriptor: Dict[str, Dict[str, str]]):
        """
        Function for initializing nodes in Bayesian Network
        descriptor: dict with types and signs of nodes
        """
        if not self.validate(descriptor=descriptor):
            if not self.type == "Hybrid" or not self.type == "Composite":
                logger_network.error(
                    f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data"
                )
                return
            else:
                logger_network.error(
                    f"Descriptor validation failed due to wrong type of column(s)."
                )
                return
        elif ["Abstract"] in self._allowed_dtypes:
            return None
        self.descriptor = descriptor
        worker_1 = builders.builders_base.VerticesDefiner(descriptor, regressor=None)

        # first stage
        worker_1.skeleton["V"] = worker_1.vertices
        # second stage
        worker_1.overwrite_vertex(
            has_logit=False,
            use_mixture=self.use_mixture,
            classifier=None,
            regressor=None,
        )
        self.nodes = worker_1.vertices

    def add_edges(
        self,
        data: pd.DataFrame,
        scoring_function: Union[Tuple[str, Callable], Tuple[str]] = ("K2", K2Score),
        progress_bar: bool = True,
        classifier: Optional[object] = None,
        regressor: Optional[object] = None,
        params: Optional[ParamDict] = None,
        optimizer: str = "HC",
        **kwargs,
    ):
        """
        Base function for Structure learning
        scoring_function: tuple with the following format (NAME, scoring_function) or (NAME, )
        Params:
        init_edges: list of tuples, a graph to start learning with
        remove_init_edges: allows changes in a model defined by user
        white_list: list of allowed edges
        """
        if not self.has_logit and check_utils.is_model(classifier):
            logger_network.error("Classifiers dict with use_logit=False is forbidden.")
            return None

        # params validation
        if params:
            # init_edges validation
            if not self.has_logit and "init_edges" in params.keys():
                type_map = np.array(
                    [
                        [
                            self.descriptor["types"][node1],
                            self.descriptor["types"][node2],
                        ]
                        for node1, node2 in params["init_edges"]
                    ]
                )
                failed = (type_map[:, 0] == "cont") & (
                    (type_map[:, 1] == "disc") | (type_map[:, 1] == "disc_num")
                )
                if sum(failed):
                    logger_network.warning(
                        f"Edges between continuous nodes and disc nodes are forbidden (has_logit = {self.has_logit}), "
                        f"they will be ignored. Indexes: {np.where(failed)[0]}"
                    )
                    params["init_edges"] = [
                        params["init_edges"][i]
                        for i in range(len(params["init_edges"]))
                        if i not in np.where(failed)[0]
                    ]

        if not self.validate(descriptor=self.descriptor):
            logger_network.error(
                f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data"
            )
            return None
        if optimizer == "HC":
            worker = HCStructureBuilder(
                data=data,
                descriptor=self.descriptor,
                scoring_function=scoring_function,
                has_logit=self.has_logit,
                use_mixture=self.use_mixture,
                regressor=regressor,
            )
        elif optimizer == "Evo":
            worker = EvoStructureBuilder(
                data=data,
                descriptor=self.descriptor,
                has_logit=self.has_logit,
                use_mixture=self.use_mixture,
                regressor=regressor,
            )
        else:
            logger_network.error(f"Optimizer {optimizer} is not supported")
            return None

        self.sf_name = scoring_function[0]

        worker.build(
            data=data,
            params=params,
            classifier=classifier,
            regressor=regressor,
            progress_bar=progress_bar,
            **kwargs,
        )

        # update family
        self.nodes = worker.skeleton["V"]
        self.edges = worker.skeleton["E"]

    def calculate_weights(self, discretized_data: pd.DataFrame):
        """
        Provide calculation of link strength according mutual information between node and its parent(-s) values.
        """
        import bamt.utils.GraphUtils as gru

        data_descriptor = gru.nodes_types(discretized_data)
        if not all([i in ["disc", "disc_num"] for i in data_descriptor.values()]):
            logger_network.error(
                f"calculate_weghts() method deals only with discrete data. Continuous data: "
                + f"{[col for col, type in data_descriptor.items() if type not in ['disc', 'disc_num']]}"
            )
        if not self.edges:
            logger_network.error(
                "Bayesian Network hasn't fitted yet. Please add edges with add_edges() method"
            )
        if not self.nodes:
            logger_network.error(
                "Bayesian Network hasn't fitted yet. Please add nodes with add_nodes() method"
            )
        weights = dict()

        for node in self.nodes:
            parents = node.cont_parents + node.disc_parents
            if parents is None:
                continue
            y = discretized_data[node.name].values
            if len(parents) == 1:
                x = discretized_data[parents[0]].values
                ls_true = information_mutual(X=y, Y=x)
                entropy_value = entropy(X=y)
                weight = ls_true / entropy_value
                weights[(parents[0], node.name)] = weight
            else:
                for parent_node in parents:
                    x = discretized_data[parent_node].values
                    other_parents = [tmp for tmp in parents if tmp != parent_node]
                    z = list()
                    for other_parent in other_parents:
                        z.append(list(discretized_data[other_parent].values))
                    ls_true = np.average(
                        information_mutual_conditional(
                            x=y, y=x, z=z, cartesian_product=True
                        )
                    )
                    entropy_value = (
                        np.average(
                            entropy_conditional(X=y, Y=z, cartesian_product=True)
                        )
                        + 1e-8
                    )
                    weight = ls_true / entropy_value
                    weights[(parent_node, node.name)] = weight
        self.weights = weights

    def set_nodes(self, nodes: List, info: Optional[Dict] = None):
        """
        additional function to set nodes manually. User should be aware that
        nodes must be a subclass of BaseNode.
        Params:
            nodes: dict with name and node (if a lot of nodes should be added)
            info: descriptor
        """
        if not info and not self.descriptor["types"]:
            logger_network.error(
                "In case of manual setting nodes user should set map for them as well."
            )
            return
        self.nodes = []
        for node in nodes:
            if issubclass(type(node), BaseNode):
                self.nodes.append(node)
            else:
                logger_network.error(f"{node} is not an instance of {BaseNode}")

        if info:
            self.descriptor = info

    def set_edges(self, edges: Optional[List[Sequence[str]]] = None):
        """
        additional function to set edges manually. User should be aware that
        nodes must be a subclass of BaseNode.
        param: edges dict with name and node (if a lot of nodes should be added)
        """

        if not self.nodes:
            logger_network.error("Graph without nodes")
        self.edges = []
        for node1, node2 in edges:
            if isinstance(node1, str) and isinstance(node2, str):
                if self[node1] and self[node2]:
                    if (
                        not self.has_logit
                        and self.descriptor["types"][node1] == "cont"
                        and self.descriptor["types"][node2] == "disc"
                    ):
                        logger_network.warning(
                            f"Restricted edge detected (has_logit=False) : [{node1}, {node2}]"
                        )
                        continue
                    else:
                        self.edges.append((node1, node2))
                else:
                    logger_network.error(f"Unknown nodes : [{node1}, {node2}]")
                    continue
            else:
                logger_network.error(
                    f"Unknown node(s) type: [{node1.__class__}, {node2.__class__}]"
                )
                continue
        self.update_descriptor()

    def set_structure(
        self,
        info: Optional[Dict] = None,
        nodes: Optional[List] = None,
        edges: Optional[List[Sequence[str]]] = None,
    ):
        """
        Function to set structure manually
        info: Descriptor
        nodes, edges:
        """
        if nodes and (info or (self.descriptor["types"] and self.descriptor["signs"])):
            if (
                any("mixture" in node.type.lower() for node in nodes)
                and not self.use_mixture
            ):
                logger_network.error("Compatibility error: use mixture.")
                return
            if (
                any("logit" in node.type.lower() for node in nodes)
                and not self.has_logit
            ):
                logger_network.error("Compatibility error: has logit.")
                return
            self.set_nodes(nodes=nodes, info=info)
        if isinstance(edges, list):
            if not self.nodes:
                logger_network.error("Nodes/info detection failed.")
                return

            builder = builders.builders_base.VerticesDefiner(
                descriptor=self.descriptor, regressor=None
            )  # init worker
            builder.skeleton["V"] = builder.vertices  # 1 stage
            if len(edges) != 0:
                # set edges and register members
                self.set_edges(edges=edges)
                builder.skeleton["E"] = self.edges
                builder.get_family()

            builder.overwrite_vertex(
                has_logit=self.has_logit,
                use_mixture=self.use_mixture,
                classifier=None,
                regressor=None,
            )

            self.set_nodes(nodes=builder.skeleton["V"])

    def _param_validation(self, params: Dict[str, Any]) -> bool:
        if all(self[i] for i in params.keys()):
            for name, info in params.items():
                try:
                    self[name].choose(node_info=info, pvals=[])
                except Exception as ex:
                    logger_network.error("Validation failed", exc_info=ex)
                    return False
            return True
        else:
            logger_network.error("Param validation failed due to unknown nodes.")
            return False

    def set_parameters(self, parameters: Dict):
        if not self.nodes:
            logger_network.error("Failed on search of BN's nodes.")

        self.distributions = parameters

        for node, data in self.distributions.items():
            if "hybcprob" in data.keys():
                if "mixture" in self[node].type.lower():
                    continue
                else:
                    if "gaussian" in self[node].type.lower():
                        model_type = "regressor"
                    else:
                        model_type = "classifier"

                    model = None
                    for v in data["hybcprob"].values():
                        if v[model_type]:
                            model = v[model_type]
                            break
                        else:
                            continue
                    if not model:
                        logger_network.warning(
                            f"Classifier/regressor for {node} hadn't been used."
                        )

                    self[node].type = re.sub(
                        r"\([\s\S]*\)", f"({model})", self[node].type
                    )
            else:
                if data.get("serialization", False):
                    regressor = data.get("regressor", None)
                    classifier = data.get("classifier", None)
                    self[node].type = re.sub(
                        r"\([\s\S]*\)", f"({regressor or classifier})", self[node].type
                    )
                else:
                    self[node].type = re.sub(
                        r"\([\s\S]*\)", f"({None})", self[node].type
                    )

    @staticmethod
    def _save_to_file(outdir: str, data: Union[dict, list]):
        """
        Function to save data to json file
        :param outdir: output directory
        :param data: dictionary to be saved
        """
        if not outdir.endswith(".json"):
            raise TypeError(
                f"Unappropriated file format. Expected: .json. Got: {path.splitext(outdir)[-1]}"
            )
        with open(outdir, "w+") as out:
            json.dump(data, out)
        return True

    def save_params(self, outdir: str):
        """
        Function to save BN params to json file
        outdir: output directory
        """
        return self._save_to_file(outdir, self.distributions)

    def save_structure(self, outdir: str):
        """
        Function to save BN edges to json file
        outdir: output directory
        """
        return self._save_to_file(outdir, self.edges)

    def save(self, bn_name, models_dir: str = "models_dir"):
        """
        Function to save the whole BN to json file.

        :param bn_name: unique name of bn user want to save. It will be used as file name (e.g. bn_name.json).
        :param models_dir: if picklization is broken, joblib will serialize models in compressed files
        in models directory.

        :return: saving status.
        """
        distributions = deepcopy(self.distributions)
        new_weights = {str(key): self.weights[key] for key in self.weights}

        to_serialize = {}
        # separate logit and gaussian nodes from distributions to serialize bn's models
        for node_name in distributions.keys():
            if "Mixture" in self[node_name].type:
                continue
            if self[node_name].type.startswith("Gaussian"):
                if not distributions[node_name]["regressor"]:
                    continue
            if (
                "Gaussian" in self[node_name].type
                or "Logit" in self[node_name].type
                or "ConditionalLogit" in self[node_name].type
            ):
                to_serialize[node_name] = [
                    self[node_name].type,
                    distributions[node_name],
                ]

        serializer = serialization_utils.ModelsSerializer(
            bn_name=bn_name, models_dir=models_dir
        )
        serialized_dist = serializer.serialize(to_serialize)

        for serialized_node in serialized_dist.keys():
            distributions[serialized_node] = serialized_dist[serialized_node]

        outdict = {
            "info": self.descriptor,
            "edges": self.edges,
            "parameters": distributions,
            "weights": new_weights,
        }
        return self._save_to_file(f"{bn_name}.json", outdict)

    def load(self, input_data: Union[str, Dict], models_dir: str = "/"):
        """
        Function to load the whole BN from json file.
        :param input_data: input path to json file with bn.
        :param models_dir: directory with models.

        :return: loading status.
        """
        if isinstance(input_data, str):
            with open(input_data) as f:
                input_dict = json.load(f)
        elif isinstance(input_data, dict):
            input_dict = deepcopy(input_data)
        else:
            logger_network.error(f"Unknown input type: {type(input_data)}")
            return

        self.add_nodes(input_dict["info"])
        self.set_structure(edges=input_dict["edges"])

        # check compatibility with father network.
        if not self.use_mixture:
            for node_data in input_dict["parameters"].values():
                if "hybcprob" not in node_data.keys():
                    continue
                else:
                    # Since we don't have information about types of nodes, we
                    # should derive it from parameters.
                    if any(
                        list(node_keys.keys()) == ["covars", "mean", "coef"]
                        for node_keys in node_data["hybcprob"].values()
                    ):
                        logger_network.error(
                            f"This crucial parameter is not the same as father's parameter: use_mixture."
                        )
                        return

        # check if edges before and after are the same.They can be different in
        # the case when user sets forbidden edges.
        if not self.has_logit:
            if not all(
                edges_before == [edges_after[0], edges_after[1]]
                for edges_before, edges_after in zip(input_dict["edges"], self.edges)
            ):
                logger_network.error(
                    f"This crucial parameter is not the same as father's parameter: has_logit."
                )
                return

        deserializer = serialization_utils.Deserializer(models_dir)

        to_deserialize = {}
        # separate logit and gaussian nodes from distributions to deserialize bn's models
        for node_name in input_dict["parameters"].keys():
            if "Mixture" in self[node_name].type:
                continue
            if self[node_name].type.startswith("Gaussian"):
                if not input_dict["parameters"][node_name]["regressor"]:
                    continue

            if (
                "Gaussian" in self[node_name].type
                or "Logit" in self[node_name].type
                or "ConditionalLogit" in self[node_name].type
            ):
                if input_dict["parameters"][node_name].get("serialization", False):
                    to_deserialize[node_name] = [
                        self[node_name].type,
                        input_dict["parameters"][node_name],
                    ]
                elif "hybcprob" in input_dict["parameters"][node_name].keys():
                    to_deserialize[node_name] = [
                        self[node_name].type,
                        input_dict["parameters"][node_name],
                    ]
                else:
                    continue

        deserialized_parameters = deserializer.apply(to_deserialize)
        distributions = input_dict["parameters"].copy()

        for serialized_node in deserialized_parameters.keys():
            distributions[serialized_node] = deserialized_parameters[serialized_node]

        self.set_parameters(parameters=distributions)

        if input_dict.get("weights", False):
            str_keys = list(input_dict["weights"].keys())
            tuple_keys = [eval(key) for key in str_keys]
            weights = {}
            for tuple_key in tuple_keys:
                weights[tuple_key] = input_dict["weights"][str(tuple_key)]
            self.weights = weights
        return True

    def fit_parameters(self, data: pd.DataFrame, n_jobs: int = 1):
        """
        Base function for parameter learning
        """
        data = data.copy()
        if data.isnull().values.any():
            logger_network.error("Dataframe contains NaNs.")
            return

        if type(self).__name__ == "CompositeBN":
            data = self._encode_categorical_data(data)

        # Turn all discrete values to str for learning algorithm
        if "disc_num" in self.descriptor["types"].values():
            columns_names = [
                name
                for name, t in self.descriptor["types"].items()
                if t in ["disc_num"]
            ]
            data[columns_names] = data.loc[:, columns_names].astype("str")

        def worker(node):
            return node.fit_parameters(data)

        results = Parallel(n_jobs=n_jobs)(delayed(worker)(node) for node in self.nodes)

        # code for debugging, do not remove
        # results = [worker(node) for node in self.nodes]

        for result, node in zip(results, self.nodes):
            self.distributions[node.name] = result

    def get_info(self, as_df: bool = True) -> Optional[pd.DataFrame]:
        """Return a table with name, type, parents_type, parents_names"""
        return get_info_(self, as_df)

    def sample(
        self,
        n: int,
        models_dir: Optional[str] = None,
        progress_bar: bool = True,
        evidence: Optional[Dict[str, Union[str, int, float]]] = None,
        as_df: bool = True,
        predict: bool = False,
        parall_count: int = 1,
        filter_neg: bool = True,
        seed: Optional[int] = None,
    ) -> Union[None, pd.DataFrame, List[Dict[str, Union[str, int, float]]]]:
        """
        Sampling from Bayesian Network
        n: int number of samples
        evidence: values for nodes from user
        parall_count: number of threads. Defaults to 1.
        filter_neg: either filter negative vals or not.
        seed: seed value to use for random number generator
        """
        from joblib import Parallel, delayed

        random.seed(seed)
        np.random.seed(seed)

        if not self.distributions.items():
            logger_network.error(
                "Parameter learning wasn't done. Call fit_parameters method"
            )
            return None
        if evidence:
            for node in self.nodes:
                if (node.type == "Discrete") & (node.name in evidence.keys()):
                    if not (isinstance(evidence[node.name], str)):
                        evidence[node.name] = str(int(evidence[node.name]))

        def wrapper():
            output = {}
            for node in self.nodes:
                parents = node.cont_parents + node.disc_parents
                if evidence and node.name in evidence.keys():
                    output[node.name] = evidence[node.name]
                else:
                    if not parents:
                        pvals = None
                    else:
                        if self.type == "Discrete":
                            pvals = [str(output[t]) for t in parents]
                        else:
                            pvals = [output[t] for t in parents]

                        # If any nan from parents, sampling from node blocked.
                        if any(pd.isnull(pvalue) for pvalue in pvals):
                            output[node.name] = np.nan
                            continue
                    node_data = self.distributions[node.name]
                    if models_dir and ("hybcprob" in node_data.keys()):
                        for obj, obj_data in node_data["hybcprob"].items():
                            if "serialization" in obj_data.keys():
                                if "gaussian" in node.type.lower():
                                    model_type = "regressor"
                                else:
                                    model_type = "classifier"
                                if (
                                    obj_data["serialization"] == "joblib"
                                    and obj_data[f"{model_type}_obj"]
                                ):
                                    new_path = (
                                        models_dir
                                        + f"\\{node.name.replace(' ', '_')}\\{obj}.joblib.compressed"
                                    )
                                    node_data["hybcprob"][obj][
                                        f"{model_type}_obj"
                                    ] = new_path
                    if predict:
                        output[node.name] = node.predict(node_data, pvals=pvals)
                    else:
                        output[node.name] = node.choose(node_data, pvals=pvals)
            return output

        if predict:
            seq = []
            for _ in tqdm(range(n), position=0, leave=True):
                result = wrapper()
                seq.append(result)
        elif progress_bar:
            seq = Parallel(n_jobs=parall_count)(
                delayed(wrapper)() for _ in tqdm(range(n), position=0, leave=True)
            )
        else:
            seq = Parallel(n_jobs=parall_count)(delayed(wrapper)() for _ in range(n))

        # code for debugging, don't remove
        # seq = []
        # for _ in tqdm(range(n), position=0, leave=True):
        #     result = wrapper()
        #     seq.append(result)

        seq_df = pd.DataFrame.from_dict(seq, orient="columns")
        seq_df.dropna(inplace=True)
        cont_nodes = [
            c.name
            for c in self.nodes
            if type(c).__name__
            not in (
                "DiscreteNode",
                "LogitNode",
                "CompositeDiscreteNode",
                "ConditionalLogitNode",
            )
        ]
        positive_columns = [
            c for c in cont_nodes if self.descriptor["signs"][c] == "pos"
        ]
        if filter_neg:
            seq_df = seq_df[(seq_df[positive_columns] >= 0).all(axis=1)]
            seq_df.reset_index(inplace=True, drop=True)

        seq = seq_df.to_dict("records")
        sample_output = pd.DataFrame.from_dict(seq, orient="columns")

        if as_df:
            if type(self).__name__ == "CompositeBN":
                sample_output = self._decode_categorical_data(sample_output)
            return sample_output
        else:
            return seq

    def get_node_by_name(self, name):
        """
        Returns the node object given its name.

        :param name: The name of the node.
        :return: Node object if found, None otherwise.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def topologically_sorted_nodes(self):
        from collections import defaultdict, deque

        # Create a dictionary to keep track of the in-degrees of each node
        in_degree = {node.name: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge[1]] += 1

        # Create a dictionary to keep track of nodes that each node is connected to
        graph = defaultdict(list)
        for edge in self.edges:
            graph[edge[0]].append(edge[1])

        # Queue for nodes with in-degree of 0
        queue = deque([node for node in self.nodes if in_degree[node.name] == 0])

        sorted_nodes = []
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)

            # Decrease the in-degree of neighboring nodes and add to queue if in-degree becomes 0
            for neighbor in graph[node.name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(self.get_node_by_name(neighbor))

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError(
                "The network has a cycle, and thus cannot be topologically sorted"
            )

        return sorted_nodes

    def vectorized_sample(
        self,
        n: int,
        batch_size: int,
        models_dir: Optional[str] = None,
        evidence: Optional[Dict[str, Union[str, int, float]]] = None,
        as_df: bool = True,
        predict: bool = False,
        filter_neg: bool = True,
    ) -> Union[None, pd.DataFrame, List[Dict[str, Union[str, int, float]]]]:
        """
        Vectorized sampling from Bayesian Network with batch processing.
        n: number of samples
        batch_size: size of each batch for sampling
        evidence: values for nodes from user
        filter_neg: either filter negative vals or not.
        """
        random.seed()
        if not self.distributions.items():
            logger_network.error(
                "Parameter learning wasn't done. Call fit_parameters method"
            )
            return None

        sample_output_dfs = []  # List to hold batches of sample outputs

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            current_batch_size = batch_end - batch_start
            batch_output = {}

            for node in self.topologically_sorted_nodes():
                parents = node.cont_parents + node.disc_parents
                if evidence is not None and node.name in evidence:
                    batch_output[node.name] = np.full(
                        current_batch_size, evidence[node.name]
                    )
                else:
                    if not parents:
                        pvals = None
                    else:
                        pvals = np.column_stack(
                            [
                                batch_output[parent][:current_batch_size]
                                for parent in parents
                            ]
                        )

                    node_data = self.distributions[node.name]
                    if predict:
                        batch_output[node.name] = node.vectorized_predict(
                            node_data, pvals, current_batch_size
                        )
                    else:
                        batch_output[node.name] = node.vectorized_choose(
                            node_data, pvals, current_batch_size
                        )

            # Convert batch output to DataFrame and store
            batch_output_df = pd.DataFrame.from_dict(batch_output)
            sample_output_dfs.append(batch_output_df)

        # Concatenate all batch DataFrames
        sample_output_df = pd.concat(sample_output_dfs, ignore_index=True)
        # Filter negative values if required
        if filter_neg:
            positive_columns = [
                node.name
                for node in self.nodes
                if self.descriptor["signs"].get(node.name) == "pos"
            ]
            sample_output_df = sample_output_df[
                (sample_output_df[positive_columns] >= 0).all(axis=1)
            ]
            sample_output_df.reset_index(drop=True, inplace=True)

        return sample_output_df if as_df else sample_output_df.to_dict("records")

    def predict(
        self,
        test: pd.DataFrame,
        parall_count: int = 1,
        progress_bar: bool = True,
        models_dir: Optional[str] = None,
    ) -> Dict[str, Union[List[str], List[int], List[float]]]:
        """
        Function to predict columns from given data.
        Note that train data and test data must have different columns.
        Both train and test datasets must be cleaned from NaNs.

        Args:
            test (pd.DataFrame): test dataset
            parall_count (int, optional):number of threads. Defaults to 1.
            progress_bar: verbose mode.

        Returns:
            predicted data (dict): dict with column as key and predicted data as value
        """
        if test.isnull().any().any():
            logger_network.error("Test data contains NaN values.")
            return {}

        from joblib import Parallel, delayed

        def wrapper(bn, test: pd.DataFrame, columns: List[str], models_dir: str):
            preds = {column_name: list() for column_name in columns}

            if len(test) == 1:
                for i in range(test.shape[0]):
                    test_row = dict(test.iloc[i, :])
                    for n, key in enumerate(columns):
                        try:
                            sample = bn.sample(
                                1,
                                evidence=test_row,
                                predict=True,
                                progress_bar=False,
                                models_dir=models_dir,
                            )
                            if sample.empty:
                                preds[key].append(np.nan)
                                continue
                            if bn.descriptor["types"][key] == "cont":
                                if (bn.descriptor["signs"][key] == "pos") & (
                                    sample.loc[0, key] < 0
                                ):
                                    # preds[key].append(np.nan)
                                    preds[key].append(0)
                                else:
                                    preds[key].append(sample.loc[0, key])
                            else:
                                preds[key].append(sample.loc[0, key])
                        except Exception as ex:
                            logger_network.error(ex)
                            preds[key].append(np.nan)
                return preds
            else:
                logger_network.error("Wrapper for one row from pandas.DataFrame")
                return {}

        columns = list(set(self.nodes_names) - set(test.columns.to_list()))
        if not columns:
            logger_network.error("Test data is the same as train.")
            return {}

        preds = {column_name: list() for column_name in columns}

        if progress_bar:
            processed_list = Parallel(n_jobs=parall_count)(
                delayed(wrapper)(self, test.loc[[i]], columns, models_dir)
                for i in tqdm(test.index, position=0, leave=True)
            )
        else:
            processed_list = Parallel(n_jobs=parall_count)(
                delayed(wrapper)(self, test.loc[[i]], columns, models_dir)
                for i in test.index
            )

        for i in range(test.shape[0]):
            curr_pred = processed_list[i]
            for n, key in enumerate(columns):
                preds[key].append(curr_pred[key][0])

        return preds

    def set_classifiers(self, classifiers: Dict[str, object]):
        """
        Set classifiers for logit nodes.
        classifiers: dict with node_name and Classifier
        """
        if not self.has_logit:
            logger_network.error("Logit nodes are forbidden.")
            return None

        for node in self.nodes:
            if "Logit" in node.type:
                if node.name in classifiers.keys():
                    node.classifier = classifiers[node.name]
                    node.type = re.sub(
                        r"\([\s\S]*\)", f"({type(node.classifier).__name__})", node.type
                    )
                else:
                    continue

    def set_regressor(self, regressors: Dict[str, object]):
        """
        Set classifiers for gaussian nodes.
        classifiers: dict with node_name and Classifier
        """

        for node in self.nodes:
            if "Gaussian" in node.type:
                if node.name in regressors.keys():
                    node.regressor = regressors[node.name]
                    node.type = re.sub(
                        r"\([\s\S]*\)", f"({type(node.regressor).__name__})", node.type
                    )
                else:
                    continue

    def plot(self, output: str):
        """
        Visualize a Bayesian Network. Result will be saved
        in the parent directory in folder visualization_result.
        output: str name of output file
        """
        return plot_(output, self.nodes, self.edges)

    def markov_blanket(self, node_name, plot_to: Optional[str] = None):
        """
        A method to get markov blanket of a node.
        :param node_name: name of node
        :param plot_to: directory to plot graph, the file must have html extension.

        :return structure: json with {"nodes": [...], "edges": [...]}
        """
        structure = GraphUtils.GraphAnalyzer(self).markov_blanket(node_name)
        if plot_to:
            plot_(
                plot_to, [self[name] for name in structure["nodes"]], structure["edges"]
            )
        return structure

    def find_family(
        self,
        node_name: str,
        height: int = 1,
        depth: int = 1,
        with_nodes: Optional[List] = None,
        plot_to: Optional[str] = None,
    ):
        """
        A method to get markov blanket of a node.
        :param node_name: name of node
        :param height: a number of layers up to node (its parents) that will be taken
        :param depth: a number of layers down to node (its children) that will be taken
        :param with_nodes: include nodes in return
        :param plot_to: directory to plot graph, the file must have html extension.

        :return structure: json with {"nodes": [...], "edges": [...]}
        """
        structure = GraphUtils.GraphAnalyzer(self).find_family(
            node_name, height, depth, with_nodes
        )

        if plot_to:
            plot_(
                plot_to, [self[name] for name in structure["nodes"]], structure["edges"]
            )

    def fill_gaps(self, df: pd.DataFrame, **kwargs):
        """
        Fill NaNs with sampled values.

        :param df: dataframe with NaNs
        :param kwargs: the same params as bn.predict

        :return df, failed: filled DataFrame and list of failed rows (sometimes predict can return np.nan)
        """
        if not self.distributions:
            logger_network.error("To call this method you must train parameters.")

        # create a mimic row to get a dataframe from iloc method
        list = [np.nan for _ in range(df.shape[1])]
        df.loc["mimic"] = list

        def fill_row(df, i):
            row = df.iloc[[i, -1], :].drop(["mimic"], axis=0)

            evidences = row.dropna(axis=1)

            return row.index[0], self.predict(evidences, progress_bar=False, **kwargs)

        failed = []
        for index in range(df.shape[0] - 1):
            if df.iloc[index].isna().any():
                true_pos, result = fill_row(df, index)
                if any(pd.isna(v[0]) for v in result.values()):
                    failed.append(true_pos)
                    continue
                else:
                    for column, value in result.items():
                        df.loc[true_pos, column] = value[0]
            else:
                continue
        df.drop(failed, inplace=True)
        return df.drop(["mimic"]), failed

    def get_dist(self, node_name: str, pvals: Optional[dict] = None):
        """
        Get a distribution from node with known parent values (conditional distribution).

        :param node_name: name of node
        :param pvals: parent values
        """
        if not self.distributions:
            logger_network.error("Empty parameters. Call fit_params first.")
            return
        node = self[node_name]

        parents = node.cont_parents + node.disc_parents
        if not parents:
            return self.distributions[node_name]

        pvals = [pvals[parent] for parent in parents]

        return node.get_dist(node_info=self.distributions[node_name], pvals=pvals)

    def _encode_categorical_data(self, data):
        for column in data.select_dtypes(include=["object", "string"]).columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            self.encoders[column] = encoder
        return data

    def _decode_categorical_data(self, data):
        data = data.apply(
            lambda col: pd.to_numeric(col).astype(int) if col.dtype == "object" else col
        )
        for column, encoder in self.encoders.items():
            data[column] = encoder.inverse_transform(data[column])
        return data
