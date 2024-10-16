import concurrent.futures
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import numpy as np

### NMF Embedding
def NMF_Embedding(X,k):
    nmf = NMF(n_components=k, init='random', random_state=0) 
    X_nmf = nmf.fit_transform(X)
    return X_nmf

### Virtual Gene Suppression Preprocessing for NMF
class VGS_NMF:
    def __init__(self, X, n_clusters, n_components):
        self.X = X  # Gene expression matrix (genes x samples)
        self.n_clusters = n_clusters  
        self.n_components = n_components  
        self.model = None

    def transform_cell(self, args):
        X_suppressed, cell_index = args
        cell = X_suppressed[:, cell_index]
        emb = self.model.transform(cell.reshape(1, -1))
        return emb

    def fit_transform(self, X_suppressed):
        self.model = NMF(n_components=self.n_components, init='random', random_state=0)
        self.model.fit(X_suppressed.T)
        cell_indices = range(X_suppressed.shape[1])
        args = [(X_suppressed, cell_index) for cell_index in cell_indices]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            VGS_X_i = list(executor.map(self.transform_cell, args))
        
        return np.asarray(VGS_X_i)

    def cluster_genes(self):
        # Cluster the genes using KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1)
        clusters = kmeans.fit_predict(self.X)
        return clusters

    def generate_features(self):
        clusters = self.cluster_genes()

        feats = []

        for cluster_label in range(self.n_clusters):
            X_suppressed = self.X.copy()
            gene_indices = np.where(clusters == cluster_label)[0]
            X_suppressed[gene_indices, :] = 0
            vgs_result = self.fit_transform(X_suppressed)
            feats.append(vgs_result)
        
        feats = np.asarray(feats)
        shape = feats.shape
        feats = feats.reshape(shape[0], shape[1], shape[3])
        feats = feats.transpose(1, 0, 2).reshape(shape[1], shape[0] * shape[3])
        return feats
