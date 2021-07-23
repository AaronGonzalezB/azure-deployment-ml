from sklearn.cluster import KMeans
import numpy as np

class Cluster(KMeans):
    def __init__(self):
        super().__init__()

        self.model = KMeans(init="k-means++", n_clusters=3, random_state=15)
        #self.labels = KMeans.labels_

    def fit(self, X):
        self.model = self.model.fit(X)
        
    def predict(self, X):
        pred = self.model.predict(X)
        print(pred)
        return pred
    

    

