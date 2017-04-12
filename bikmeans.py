import math
import numpy as np
from scipy.spatial.distance import cdist
import random
from dotdict import dotdict

#from sklearn.cluster import KMeans
try:
  from itertools import izip as zip
except ImportError: # will be 3.x series
  pass

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
class KMeans:
    def __init__(self, k = 2, delta=.001, maxiter=300, metric='euclid'):
        self.k = k
        self.delta = delta
        self.maxiter = maxiter
        self.metric = metric
    '''
    def dot_product(self,v1, v2):
        return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

    def cosine_measure(self, v1, v2):
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return prod / (len1 * len2)    

    
    '''

    '''
    #machinelearning cơ bản   
    
    def kmeans_assign_labels(self, X, centers):
        # calculate pairwise distances btw data and center by metric measure
        D = cdist(X, centers, self.metric)
	    # return index of the closest center
        return np.argmin(D, axis = 1)

    
    def kmeans_update_centers(self, X, labels):
        centers = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            # collect all points assigned to the k-th cluster 
            Xk = X[labels == k, :]
            # take average
            centers[k,:] = np.mean(Xk, axis = 0)
        return centers

    def kmeans(self, X):
        centers = [self.init_random_cluster(X)]
        N, dim = X.shape
        k, cdim, e = np.array(centers).shape
        if dim != cdim:
            raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
                X.shape, centers.shape ))
        new_centers = [self.init_random_cluster(X)]
        labels = []
        it = 0
        while True:
            labels.append(self.kmeans_assign_labels(X, centers[-1]))
            new_centers = self.kmeans_update_centers(X, labels[-1])
            if self.has_converged(iterations=it, new_centers=new_centers, old_centers=centers[-1]):
                break
            centers.append(new_centers)
            it += 1
        return (centers, labels, it)

        '''
    #kmeans với giải thuật Lloyd’s

    def init_random_cluster(self, X):
    	#random choice k centroids
        return X[np.random.choice(X.shape[0], self.k, replace=False)]

    def has_converged(self, iterations, new_centers, old_centers):
        if self.maxiter == iterations:
           return True
        return (set([tuple(a) for a in old_centers]) == set([tuple(a) for a in new_centers]))

    def assign_point_to_clusters(self, X, centroids):
        clusters = {}
        for x in X:
            D = cdist([x], centroids, self.metric)
            index = np.argmin(D, axis=1)
            try: 
                clusters[index[0]].append(x)
            except KeyError:
                clusters[index[0]] = [x]
        
        return clusters
    
    def recalculate_centroids(self, clusters):

        centroids = []
        keys = sorted(clusters.keys())
        for k in keys:
            centroids.append(np.mean(clusters[k], axis = 0))
        return centroids
    
    def kmeans(self, X):
        old_centroids = random.sample(list(X), 2)
        centroids = random.sample(list(X), 2)
        iterations = 0
        while not self.has_converged(iterations, centroids, old_centroids):
            old_centroids = centroids
            lc_clusters = self.assign_point_to_clusters(X = X, centroids = centroids)
            centroids = self.recalculate_centroids(lc_clusters)
            iterations +=1

        final_clusters = []
        
        for index in range(0, len(centroids)):
            cluster = dotdict()
            cluster.centroid = centroids[index]
            cluster.vectors = lc_clusters[index]
            final_clusters.append(cluster)

        return final_clusters
        
    def similitary(self, clusters):
        sim = 0.0
        for cluster in clusters:
            centroid = cluster['centroid']
            for index in range(len(centroid)):
                sim += centroid[index]**2
        
        return sim / len(clusters)

class BiKMeans(KMeans):

    def __init__(self, k, max_iter = 5):
        self.k = k
        self.max_iter = max_iter
        self.clusters = []

    def dot_product(self,v1, v2):
        return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

    def cosine_measure(self, v1, v2):
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return prod / (len1 * len2)

    def execute(self, X):
        clusters = []
        cluster = dotdict()
        cluster.vectors = []
        
        for x in X:
            cluster.vectors.append(x)
        
        cluster.centroid = self.calculate_centroid(cluster.vectors)
       
        clusters.append(cluster)
        while (len(clusters) != self.k):
            split_cluster = self.find_smallest_sim_cluster(clusters)
            clusters.remove(split_cluster)
            max_cluster = float("-inf")
            max_bicluster = None
            for i in range(1, self.max_iter):
                kmeans = KMeans()
                biclusters = kmeans.kmeans(np.array(split_cluster.vectors))
                sim = kmeans.similitary(biclusters)
                if (sim > max_cluster):
                    max_bicluster = biclusters
                    max_cluster = sim

            clusters.extend(biclusters)
        return clusters


    def convert_dotdict(self, datas):
        cluster = dotdict()
        cluster.vectors = []
        cluster.vectors[0] = datas[0]
        cluster.vectors[1] = datas[1]
        cluster.centroids = self.calculate_centroid(cluster.vectors)
        return datas

    
    def calculate_centroid(self, clusters):
        '''
        centroid = []
        for cluster in clusters:
            print(len(cluster))
            for index in range(0,1):
                print(index)
                centroid.append()
                print(centroid[0])
        
        len_vectors = len(clusters)
        for index in centroid:
            centroid[index] = centroid[index]/len_vectors
            '''
        return list(np.mean(clusters, axis=0))

    def find_smallest_sim_cluster(self, clusters):
        min_sim = float("inf")
        min_cluster = None
        for cluster in clusters:
            centroid = cluster['centroid']
            sim = 0.0
            for index in range(len(centroid)):
                sim += centroid[index]**2
            if sim < min_sim:
                min_sim = sim
                min_cluster = cluster
        return min_cluster


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

#X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

#kmeans = KMeans()
#print(kmeans.kmeans(X))

bikmeans = BiKMeans(2)
print(bikmeans.execute(X))

#tính cosine distance(2) = 1 - cosine_similitary giữa các vector
#http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means