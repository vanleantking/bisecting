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

X = np.array([[1, 2], [1, 4], [5, 1],
              [4, 2], [4, 4], [4, 1], [2,3]])

Y = np.array([[-0.12300821,  0.16935564, -0.00048462, -0.09636025, -0.0113105 ,
        0.09624533,  0.17103165, -0.14667498,  0.01485234, -0.06319674,
        0.0288428 , -0.09043634, -0.0438677 ,  0.01985058,  0.08806337,
       -0.02952759,  0.063406  , -0.07027014,  0.26124352, -0.10737031,
        0.14520512, -0.01031582,  0.01940708,  0.17562549,  0.07795008,
        0.14863656, -0.01422945, -0.15023584, -0.01968571, -0.0303746 ,
       -0.0111963 , -0.03511471,  0.00798229,  0.12621938,  0.05969337,
       -0.01008857,  0.00553513,  0.04313779, -0.06076963,  0.1129121 ,
        0.17053346,  0.07341338,  0.04231231, -0.10309229,  0.11034401,
        0.02649869, -0.17654717,  0.12045622,  0.21747996, -0.0752383 ,
       -0.05141732, -0.0462486 , -0.1981267 ,  0.06977668, -0.11980595,
       -0.08442397,  0.0959678 , -0.16775822, -0.09715045,  0.14871019,
        0.11787972,  0.04212514, -0.08923571, -0.15046366, -0.16514437,
       -0.06144415,  0.02329804, -0.15019874,  0.13501469, -0.09393135,
        0.065232  ,  0.04824227, -0.0676146 ,  0.08632225, -0.16905181,
        0.02930885,  0.08388016, -0.09963868, -0.02302801,  0.10340357,
        0.04363841,  0.07550566, -0.10605617,  0.01477681,  0.0537468 ,
        0.16475934,  0.11739064, -0.1232534 ,  0.03786069,  0.02864431,
       -0.10167701, -0.09500054, -0.00132036,  0.05924389, -0.06693932,
       -0.03242361, -0.09994838, -0.09795284, -0.0078222 ,  0.00616494],
       [ 0.86902279,  0.60884368,  0.57266271,  0.07157172, -0.70431012,
       -0.03731671,  0.29461238, -0.83505875,  0.14014515,  1.10285068,
       -0.81214136,  0.81773388, -1.062783  ,  0.14835823,  0.7231673 ,
        0.51047283,  0.45346114,  0.63662982, -0.80885816,  0.24498159,
        0.51244891,  0.07006902,  0.73245466, -0.73820525, -1.99913812,
       -0.52003109, -0.74266839,  0.21672016, -0.02710354, -0.25216568,
        0.2462056 ,  0.12647945, -0.04710531,  1.18399036,  1.38778245,
        0.19668105,  0.45988256,  0.85147768,  0.19943738, -0.25728178,
        0.16558152,  1.62494695,  0.16693205,  0.37021822, -1.06122613,
       -1.17925143,  1.2622875 , -0.1917648 , -0.4060474 , -0.31460407,
        1.66843295,  0.12631953, -0.08652818,  0.34891826, -0.44256067,
        0.3213926 , -0.17454527, -0.0108836 ,  0.98308694, -0.02955383,
       -0.63041806, -0.40527099,  1.3295145 ,  0.80947334,  1.13145816,
       -0.77781767,  0.13407677,  1.26985383,  0.15022308, -0.03634179,
       -0.67176056,  1.56496489, -0.08852583,  0.70965463,  0.3344461 ,
        0.93408871,  1.5441612 ,  0.9147054 , -0.15783405, -0.32352144,
        0.70175266,  1.1756568 , -0.18386486,  0.05610284, -1.138448  ,
        0.78596282,  1.26440132,  1.77710056, -0.65983433,  0.75105411,
       -0.4184272 ,  1.27766049, -0.10292232,  0.44144502, -0.29862827,
       -0.65421659, -0.92001212,  1.19741464, -0.39841297,  1.27490616],
       [ 0.86902279,  0.60884368,  0.57266271,  0.07157172, -0.70431012,
       -0.03731671,  0.29461238, -0.83505875,  0.14014515,  1.10285068,
       -0.81214136,  0.81773388, -1.062783  ,  0.14835823,  0.7231673 ,
        0.51047283,  0.45346114,  0.63662982, -0.80885816,  0.24498159,
        0.51244891,  0.07006902,  0.73245466, -0.73820525, -1.99913812,
       -0.52003109, -0.74266839,  0.21672016, -0.02710354, -0.25216568,
        0.2462056 ,  0.12647945, -0.04710531,  1.18399036,  1.38778245,
        0.19668105,  0.45988256,  0.85147768,  0.19943738, -0.25728178,
        0.16558152,  1.62494695,  0.16693205,  0.37021822, -1.06122613,
       -1.17925143,  1.2622875 , -0.1917648 , -0.4060474 , -0.31460407,
        1.66843295,  0.12631953, -0.08652818,  0.34891826, -0.44256067,
        0.3213926 , -0.17454527, -0.0108836 ,  0.98308694, -0.02955383,
       -0.63041806, -0.40527099,  1.3295145 ,  0.80947334,  1.13145816,
       -0.77781767,  0.13407677,  1.26985383,  0.15022308, -0.03634179,
       -0.67176056,  1.56496489, -0.08852583,  0.70965463,  0.3344461 ,
        0.93408871,  1.5441612 ,  0.9147054 , -0.15783405, -0.32352144,
        0.70175266,  1.1756568 , -0.18386486,  0.05610284, -1.138448  ,
        0.78596282,  1.26440132,  1.77710056, -0.65983433,  0.75105411,
       -0.4184272 ,  1.27766049, -0.10292232,  0.44144502, -0.29862827,
       -0.65421659, -0.92001212,  1.19741464, -0.39841297,  1.27490616],
       [ 0.86902279,  0.60884368,  0.57266271,  0.07157172, -0.70431012,
       -0.03731671,  0.29461238, -0.83505875,  0.14014515,  1.10285068,
       -0.81214136,  0.81773388, -1.062783  ,  0.14835823,  0.7231673 ,
        0.51047283,  0.45346114,  0.63662982, -0.80885816,  0.24498159,
        0.51244891,  0.07006902,  0.73245466, -0.73820525, -1.99913812,
       -0.52003109, -0.74266839,  0.21672016, -0.02710354, -0.25216568,
        0.2462056 ,  0.12647945, -0.04710531,  1.18399036,  1.38778245,
        0.19668105,  0.45988256,  0.85147768,  0.19943738, -0.25728178,
        0.16558152,  1.62494695,  0.16693205,  0.37021822, -1.06122613,
       -1.17925143,  1.2622875 , -0.1917648 , -0.4060474 , -0.31460407,
        1.66843295,  0.12631953, -0.08652818,  0.34891826, -0.44256067,
        0.3213926 , -0.17454527, -0.0108836 ,  0.98308694, -0.02955383,
       -0.63041806, -0.40527099,  1.3295145 ,  0.80947334,  1.13145816,
       -0.77781767,  0.13407677,  1.26985383,  0.15022308, -0.03634179,
       -0.67176056,  1.56496489, -0.08852583,  0.70965463,  0.3344461 ,
        0.93408871,  1.5441612 ,  0.9147054 , -0.15783405, -0.32352144,
        0.70175266,  1.1756568 , -0.18386486,  0.05610284, -1.138448  ,
        0.78596282,  1.26440132,  1.77710056, -0.65983433,  0.75105411,
       -0.4184272 ,  1.27766049, -0.10292232,  0.44144502, -0.29862827,
       -0.65421659, -0.92001212,  1.19741464, -0.39841297,  1.27490616],
       [ 0.86902279,  0.60884368,  0.57266271,  0.07157172, -0.70431012,
       -0.03731671,  0.29461238, -0.83505875,  0.14014515,  1.10285068,
       -0.81214136,  0.81773388, -1.062783  ,  0.14835823,  0.7231673 ,
        0.51047283,  0.45346114,  0.63662982, -0.80885816,  0.24498159,
        0.51244891,  0.07006902,  0.73245466, -0.73820525, -1.99913812,
       -0.52003109, -0.74266839,  0.21672016, -0.02710354, -0.25216568,
        0.2462056 ,  0.12647945, -0.04710531,  1.18399036,  1.38778245,
        0.19668105,  0.45988256,  0.85147768,  0.19943738, -0.25728178,
        0.16558152,  1.62494695,  0.16693205,  0.37021822, -1.06122613,
       -1.17925143,  1.2622875 , -0.1917648 , -0.4060474 , -0.31460407,
        1.66843295,  0.12631953, -0.08652818,  0.34891826, -0.44256067,
        0.3213926 , -0.17454527, -0.0108836 ,  0.98308694, -0.02955383,
       -0.63041806, -0.40527099,  1.3295145 ,  0.80947334,  1.13145816,
       -0.77781767,  0.13407677,  1.26985383,  0.15022308, -0.03634179,
       -0.67176056,  1.56496489, -0.08852583,  0.70965463,  0.3344461 ,
        0.93408871,  1.5441612 ,  0.9147054 , -0.15783405, -0.32352144,
        0.70175266,  1.1756568 , -0.18386486,  0.05610284, -1.138448  ,
        0.78596282,  1.26440132,  1.77710056, -0.65983433,  0.75105411,
       -0.4184272 ,  1.27766049, -0.10292232,  0.44144502, -0.29862827,
       -0.65421659, -0.92001212,  1.19741464, -0.39841297,  1.27490616]])
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
class KMeans:
    def __init__(self, k = 2, delta=.001, maxiter=300, metric='cosine'):
        self.k = k
        self.delta = delta
        self.maxiter = maxiter
        self.metric = metric

    def init_random_cluster(self, X):
    	#random choice k centroids
        return X[np.random.choice(X.shape[0], self.k, replace=False)]

    def has_converged(self, iterations, new_centers, old_centers):
        if self.maxiter == iterations:
           return True
        return (set([tuple(a) for a in old_centers]) == set([tuple(a) for a in new_centers]))


    def cosine_similitary(self, vector1, vector2):
        prod = 0.0
        
        mag1 = 0.0
        mag2 = 0.0
        
        for index, value in enumerate(vector1):
            prod += vector1[index] * vector2[index]
            mag1 += vector1[index] * vector1[index]
            mag2 += vector2[index] * vector2[index]
        
        return prod / (math.sqrt(mag1) * math.sqrt(mag2))

    def assign_point_to_clusters(self, X, centroids):
        clusters = {i : [] for i in range(self.k)}
        for x in X:
            D = 1 - cdist([x], centroids, self.metric)
            mean_index = min([(m[0], self.cosine_similitary(x, centroids[m[0]])) for m in enumerate(centroids)], key=lambda t: t[1])[0]

            try: 
                clusters[mean_index].append(x)
            except KeyError:
                clusters[mean_index] = [x]

        for key, cluster in clusters.items():
            if not cluster:
                cluster.append(X[np.random.randint(0, len(X), size=1)].flatten().tolist())
        return clusters
    
    def recalculate_centroids(self, clusters):

        centroids = []
        keys = sorted(clusters.keys())
        for k in keys:
            centroids.append(np.mean(clusters[k], axis = 0))
        return centroids
    
    def kmeans(self, X):
        old_centroids = X[np.random.choice(X.shape[0], size=self.k, replace=False), :]
        centroids = old_centroids
        iterations = 0
        lc_clusters = []
        while True:            
            old_centroids = centroids
            lc_clusters = self.assign_point_to_clusters(X = X, centroids = centroids)
            centroids = self.recalculate_centroids(lc_clusters)
            iterations +=1
            if (self.has_converged(iterations, centroids, old_centroids)):
                break

        final_clusters = []
        
        for index in range(len(centroids)):
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

    def __init__(self, k, max_iter = 500):
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
        while True:
            split_cluster = self.find_smallest_sim_cluster(clusters)

            # re-construct clusters except the split cluster
            clusters = [d for d in clusters if not np.array_equal(d['centroid'], split_cluster['centroid'])]
            old_clusters = clusters
            max_cluster = float("-inf")
            max_bicluster = None
            for i in range(self.max_iter):
                kmeans = KMeans(k=2)

                # loop max_iter to find the best way to split
                biclusters = kmeans.kmeans(np.array(split_cluster.vectors))
                sim = kmeans.similitary(biclusters)
                if (sim > max_cluster):
                    max_bicluster = [d for d in biclusters]
                    max_cluster = sim

            clusters.extend(biclusters)
            if (self.bi_convegence(clusters, old_clusters)):
                break
        return clusters


    def convert_dotdict(self, datas):
        cluster = dotdict()
        cluster.vectors = []
        cluster.vectors[0] = datas[0]
        cluster.vectors[1] = datas[1]
        cluster.centroids = self.calculate_centroid(cluster.vectors)
        return datas


    def bi_convegence(self, clusters, old_clusters):
        for cluster in clusters:
            if (len(cluster['vectors']) == 1):
                return True
        return (set([tuple(a['centroid']) for a in clusters]) == set([tuple(a['centroid']) for a in old_clusters]))

    
    def calculate_centroid(self, clusters):
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
# K = 3

# Y = np.array([[1,0], [4,0]])
# kmeans = KMeans()
# print(kmeans.kmeans(X,k=2))

bikmeans = BiKMeans(3)
print(bikmeans.execute(Y))

#tinh cosine distance(2) = 1 - cosine_similitary giua cac vecto
#http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means