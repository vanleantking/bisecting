import math
import numpy as np
from scipy.spatial.distance import cdist
import random

#from sklearn.cluster import KMeans
try:
  from itertools import izip as zip
except ImportError: # will be 3.x series
  pass

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
class KMeans:
    def __init__(self, k = 2, delta=.001, maxiter=300, metric='cosine'):
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
        old_centroids = self.init_random_cluster(X)
        centroids = self.init_random_cluster(X)
        iterations = 0
        while not self.has_converged(iterations, centroids, old_centroids):
            old_centroids = centroids
            lc_clusters = self.assign_point_to_clusters(X = X, centroids = centroids)
            centroids = self.recalculate_centroids(lc_clusters)
            iterations +=1

        return (centroids, lc_clusters)

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

    #def find_smallest_sim_cluster(clusters):


    def execute(self, X):
        clusters = []
        split_cluster = X
        while (len(clusters) < k):
            cluster = self.find_smallest_sim_cluster(clusters)
            clusters.remove(cluster)
            max_cluster = float("-inf")
            max_bicluster = None

            for i in range(1, self.max_iter):
                kmeans = KMeans(cluster)
                (centroids, bi_clusters) = kmeans.kmeans(cluster)
                sim = self.similitary(bi_clusters)

                if (sim > max_cluster):
                    max_bicluster = bi_clusters
                    max_cluster = sim

            clusters.append(max_cluster)
        self.clusters = clusters

    def find_smallest_sim_cluster(self, clusters):
        min_sim = float("-inf")
        min_cluster = None

        for cluster in clusters:
            sim = 0.0
            sum_cluster = np.sum()


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

#X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

kmeans = KMeans()
print(kmeans.kmeans(X))

#tính cosine distance(2) = 1 - cosine_similitary giữa các vector
#http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means