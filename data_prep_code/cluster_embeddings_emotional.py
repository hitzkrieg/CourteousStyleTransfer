import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn import metrics
from time import time 
import pickle

def unpickle_2(filename):
	with open(filename, 'rb') as fp:
		return pickle.load(fp, encoding ='bytes')


def unpickle(filename):
	with open(filename, 'rb') as fp:
		return pickle.load(fp)

# import sys
# import codecs
# sys.stdout=codecs.getwriter('utf-8')(sys.stdout)
# print(unicode_obj)


def dbscan_cluster(X, sentences, sent_inbounds):
	with open('dbscan_cluster_demo_emotional_embed.txt', 'w+', encoding = 'utf-8') as f:

		db = DBSCAN(eps=0.01, min_samples=10).fit(X)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_
		unique_labels = set(labels)

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

		f.write('\n Estimated number of clusters: %d' % n_clusters_)
		f.write("\n Silhouette Coefficient: %0.3f"
		      % metrics.silhouette_score(X, labels))

		clusters = []
		for i in range(n_clusters_+1):
			clusters.append([])
		for i in range(len(X)):
			try:
				clusters[labels[i]].append(sentences[i])
			except IndexError as e:
				f.write(i , labels[i],sentences[i] )

		for i in range(n_clusters_+1):
			f.write("\n **********")
			f.write("\n Cluster: {}. Size: {}\n".format(i, len(clusters[i])))
			f.write('\n'.join(clusters[i][:20]))



def kmeans_cluster(X, sentences, sent_inbounds):
	with open('kmeans_cluster_demo_emotional_embed.txt', 'w+', encoding = 'utf-8') as f:
		mbk = MiniBatchKMeans(init='k-means++', n_clusters=100, batch_size=100,
		                      n_init=10, max_no_improvement=10, verbose=0,
		                      random_state=0)
		t0 = time()
		mbk.fit(X)
		t_mini_batch = time() - t0
		f.write("\nTime taken to run MiniBatchKMeans {} seconds".format(t_mini_batch))
		# mbk_means_labels_unique = np.unique(mbk.labels_)
		labels = mbk.labels_
		unique_labels = set(labels)

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = 100

		f.write('\nEstimated number of clusters: %d' % n_clusters_)
		f.write("\nSilhouette Coefficient: %0.3f"
		      % metrics.silhouette_score(X, labels))

		# for i in range(30):
		# 	if sent_inbounds[i] == 'False':
		# 		f.write(sentences[i], 'label: ', labels[i])

		clusters = []
		for i in range(n_clusters_+1):
			clusters.append([])
		for i in range(len(X)):
			try:
				clusters[labels[i]].append(sentences[i])
			except IndexError as e:
				f.write(i, labels[i],sentences[i] )

		for i in range(n_clusters_+1):
			f.write("\n**********")
			f.write("\nCluster: {}. Size: {}\n".format(i, len(clusters[i])))
			f.write('\n'.join(clusters[i][:20]))

sentences = unpickle('pickle_files/sentences.pickle')[:40000]
sent_inbounds = unpickle('pickle_files/sent_inbounds.pickle')[:40000]
X = unpickle_2('pickle_files/emotional_encodings1_2.pickle')[:40000]

sentences = [sent for i,sent in enumerate(sentences) if sent_inbounds[i] == 'False']
X = [sent for i,sent in enumerate(X) if sent_inbounds[i] == 'False']
sent_inbounds = [sent for i,sent in enumerate(sent_inbounds) if sent_inbounds[i] == 'False']

kmeans_cluster(X, sentences, sent_inbounds)