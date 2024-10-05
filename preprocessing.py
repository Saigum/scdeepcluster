import scanpy as sc
import gc
import pandas as pd
import csv
import networkx as nx
import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from node2vec import Node2Vec







def preprocessing_for_scEGA(scdata):
    scdata = scdata.T
    sc.pp.pca(scdata)
    # mitochondrial genes
    scdata.var["mt"] = scdata.var_names.str.startswith("MT-")
    # ribosomal genes
    scdata.var["ribo"] = scdata.var_names.str.startswith(("RPS","RPL"))
    # hemoglobin genes.
    scdata.var["hb"] = scdata.var_names.str.contains(("^HB[^(P)]"))
    sc.pp.calculate_qc_metrics(
        scdata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )
    sc.pp.filter_cells(scdata, min_genes=3)
    sc.pp.neighbors(scdata)
    adj_list = scdata.obsp["distances"].indices.reshape(scdata.shape[0],-1)
    sc.pp.normalize_total(scdata, target_sum=1e4)
    sc.pp.log1p(scdata)
    sc.pp.highly_variable_genes(scdata, n_top_genes=2000)
    return scdata, adj_list

def get_gene_graph(dataset_name="Biase"):
    edge_list = pd.DataFrame(csv.reader
                             (open(f"{dataset_name}_hvg_string_interactions.tsv")
                             ,delimiter="\t"))
    edge_list.columns = edge_list.iloc[0]
    edge_list = edge_list.drop(0)
    gene_graph = nx.Graph()
    gene_graph.add_edges_from(edge_list[["node1_string_id","node2_string_id"]].values) 
    return gene_graph

def get_gene_matrix(gene_graph):

    node2vec = Node2Vec(gene_graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = model.wv.vectors

    return node_embeddings,model