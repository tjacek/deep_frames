import numpy as np
import utils
from sklearn.cluster import MiniBatchKMeans
from time import time

def cluster_images(in_path,out_path):
    images=utils.read_array(in_path)
    print(images.shape)
    #print(images[0].shape)
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
    in_path="/home/user/df/imgs.obj"
    out_path="/home/user/df/clust.obj"
    cluster_images(in_path,out_path)
