import numpy as np
import utils
from dim_reduction import load_data
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
from dim_reduction import DfConf
from time import time

class Clusters(object):
    def __init__(self,path):
        self.clusters=utils.read_object(path)

    def find_clusters(self,images):
        def least_distance(y):
            fun=lambda x:distance.euclidean(x,y)
            dist=map(fun,self.clusters) 
            return np.argmin(dist)
        result=map(least_distance,images)
        return result

def cluster_images(in_path,out_path):
    images=utils.read_array(in_path)
    clusters=clustering_mini_batch(images)
    utils.save_object(out_path,clusters)

def clustering_mini_batch(images,clusters=10):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, 
                      batch_size=100,n_init=10, max_no_improvement=10, 
                      verbose=0,random_state=0)
    t0 = time()
    mbk.fit(images)
    t_mini_batch = time() - t0
    print("Time taken to run clustering %0.2f seconds" % t_mini_batch)
    return mbk.cluster_centers_
 
if __name__ == "__main__":
    path="/home/user/df/exp2/"
    conf=DfConf(path)
    cluster_images(conf.vectors,conf.cls)
