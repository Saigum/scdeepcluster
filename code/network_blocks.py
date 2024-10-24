
import torch as th
import torch.nn as nn
import torch.nn.functional as F




class graphattention_layer(nn.Module):
    def __init__(self,input_size,output_size,adjM):
        '''WkH_{i-1} is of dimension : CurrentNodeShape x N'''  
        self.inpshape = input_size
        self.opshape = output_size
        self.A = adjM
        super(graphattention_layer,self).__init__()
        self.vkt = nn.Linear(in_features=output_size,out_features=  1)
        self.vkr = nn.Linear(in_features=output_size,out_features= 1)
        self.W =  nn.Linear(in_features=input_size,out_features=output_size) 
    def forward(self, H_k):
        '''H_k represents the previous layer's graph representation'''
        M_s = self.A * self.vkt(F.relu(self.W(H_k))).T
        M_r = (self.A * self.vkr(F.relu(self.W(H_k))).T).T
        Attention = F.softmax(F.sigmoid(M_s+M_r))
        H_new = Attention@F.relu(self.W(H_k))
        return H_new
    
class encoder(nn.Module):
    def __init__(self,adjM,input_embeddings):
        super(encoder,self).__init__( )
        ''' 
        remember that in pytorch, your input_size is the last dimension of your input
        So when my input is F*N, input_size = F
        also a row in my matrix corresponds to a cell's representation
        '''
       
        self.layer1 = graphattention_layer(input_size=input_embeddings
                                           ,output_size=512
                                           ,adjM=adjM)
        self.layer2 = graphattention_layer(input_size=512
                                           ,output_size=256
                                           ,adjM=adjM)
        self.layer3 = graphattention_layer(input_size=256
                                           ,output_size=64
                                           ,adjM=adjM)
    def forward(self, X):
        '''
        X here is the node embeddings, its of shape (N*embedding_size)
        I'm gonna tranpose it once in the start, and then at the end.
        H3 is of size N*64
        I'm gonna transpose it back to 64*N
        '''
        H1 = self.layer1(X)
        H2 = self.layer2(H1)
        H3 = self.layer3(H2)
        return H3
    
class decoder(nn.Module):
    def __init__(self,adjM,reconstruction_embedding,gene_embeddings):
        super(decoder,self).__init__()
        self.gGraph = gene_embeddings
        self.layer1 = graphattention_layer(input_size=64,
                                           output_size=256,adjM=adjM)
        self.layer2 = graphattention_layer(input_size=256,
                                           output_size=512,adjM=adjM)
        self.layer3 = graphattention_layer(input_size=512,
                                           output_size=reconstruction_embedding,adjM=adjM)
        self.layer3_2 = graphattention_layer(input_size=512,
                                           output_size=gene_embeddings.shape[0],adjM=adjM)
    def forward(self, H):
        '''
        H here is the encoder's output
        I'm gonna stack the gene embeddings to the H matrix
        Encoder should have returned a N* 64 matrix
        Gene embeddings should be of dimension 64* num_nodes , which was 647 for the first run.
        
        '''
        # Now its a (N)*64 matrix
        decoderPass = th.cat((self.gGraph, H.T), dim=1).T
        # now our matrix should be (647+N (N is number of nodes ))*64
        H1 = self.layer1(decoderPass)
        H2 = self.layer2(H1)
        H3= self.layer3(H2)
        ''' H3 would be of size N*N'''
        return H3 
    


class scdEGA(nn.Module):
    def __init__(self,hidden_size,Cellmatrix_pca,adjM,GeneGraph):
        '''GeneGraph is the gene embeddings from the node2vec model
           run on a PPI graph constructed from the gene interactions.
           This will be the gene matrix (so gene loss) we wish to reconstruct.
                      
           Cellmatrix_pca is the PCA reduced cell matrix.
           This will be the cell matrix (so cell loss) we wish to reconstruct.

           adjM is the adjacency matrix of the cell graph.
        '''
        super(scdEGA,self).__init__()
        self.gc = Cellmatrix_pca
        self.encoder = encoder(hidden_size,adjM,Cellmatrix_pca.shape[1])
        self.decoder = decoder(hidden_size,adjM,GeneGraph)
    
    def unsupervised_loss_unit(self):
        pass

    def forward(self, H):
        cell_embeddings = self.encoder(H)
        reconstructed_cell_matrix,reconstructed_gene_matrix = self.decoder(cell_embeddings)
        self.reconstruction_cell_loss = F.cosine_similarity(reconstructed_cell_matrix,self.gc)
        self.reconstruction_gene_loss = F.mean_absolute_error(reconstructed_gene_matrix,self.gc)
        self.selfsupervised_loss = self.reconstruction_cell_loss+self.reconstruction_gene_loss
        return reconstructed_cell_matrix,reconstructed_gene_matrix

