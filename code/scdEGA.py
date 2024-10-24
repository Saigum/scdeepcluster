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

from scdeepcluster.preprocessing import preprocessing_for_scEGA,get_gene_graph,get_gene_matrix
from scdeepcluster.network_blocks import graphattention_layer,encoder,decoder,scdEGA
from tqdm import tqdm

class Trainer(nn.Module):
    #scdata is assumed to be the scanpy object
    def __init__(self,scdata,dataset_name="Biase"):
        self.scdata = scdata.T
        self.scdata, adj_list = preprocessing_for_scEGA(self.scdata)
        adjM = th.zeros(size=(self.scdata.T.shape[1],self.scdata.T.shape[1]))
        for i in range(adj_list.shape[0]):
            for j in range(5):
                adjM[i][adj_list[i][j]] = 1
        self.adjM = adjM

        self.gene_graph = get_gene_graph(dataset_name)
        self.node2vecmodel,self.gene_embeddings = get_gene_matrix(self.gene_graph)
        
        self.model = scdEGA(hidden_size=64,
                            GeneCellMatrix=self.scdata,
                            adjM=self.adjM,
                            GeneGraph=self.gene_embeddings)
        
    def train(self,epochs=100,batch_size=32):
        ''' Batch size will be N, the size of the entire graph,
            So batching isn't needed.
        '''
        optimizer = th.optim.Adam(self.model.parameters(),lr=0.001)
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            batch = self.scdata
            optimizer.zero_grad()
            output = self.model(batch)
            loss = F.mse_loss(output,batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss : {loss.item()}")

    def get_embeddings(self):
        return self.model.encoder(self.scdata)
    def get_reconstruction(self):
        return self.model.decoder(self.get_embeddings())
    def get_gene_embeddings(self):
        return self.model.gene_embeddings
    def get_gene_graph(self):
        return self.adjM
    def get_cell_graph(self):
        return self.scdata