from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

k = 2

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)

print(kmeans.predict([[0, 0], [4, 4]]))

print(kmeans.cluster_centers_)
#print(cosine_similarity([2,3], X))
print(cosine_similarity(X))
print(np.argmin(cdist(([[2,3]]), X, 'cosine'), axis=1))