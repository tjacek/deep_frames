import sys
import numpy as np
import utils 
from reduction.autoencoder import AutoEncoder,AutoEncoderReduction,learning_autoencoder

class DfConf(object):
    def __init__(self,path):
        self.path=path
        self.nn=path+"nn"
        self.cls=path+"clust.obj"
        self.seq=path+"dataset.seq"
        self.vectors=path+"vectors.obj"

def load_data(path,batch_size=25):
    all_files=utils.get_all_files(path)
    all_files=utils.append_path(path,all_files)
    images=utils.read_images(all_files)
    images=utils.flatten_images(images)
    images=map(utils.normalize,images)
    images=np.array(images)
    n_batches=get_number_of_batches(batch_size,len(images))
    def get_batch(i):
        return images[i * batch_size: (i+1) * batch_size]
    batches=map(get_batch,range(n_batches))
    batches = [np.array(batch) for batch in batches]
    print("Dataset loaded")
    return np.array(batches)

def get_number_of_batches(batch_size,n_images):
    n_batches=n_images / batch_size
    if(n_images % batch_size != 0):
        n_batches+=1
    return n_batches

def train_autoencoder(in_path,out_path,training_epochs=15,
            learning_rate=0.1,batch_size=25):
    dataset = load_data(in_path,batch_size)
    da=learning_autoencoder(dataset,training_epochs,learning_rate,batch_size)
    utils.save_object(out_path,da)

def save_reduction(in_path,out_path,nn_path,csv=False):
    dataset=load_data(in_path,1)
    dataset=[inst for inst in dataset]
    autoencoder=AutoEncoderReduction(nn_path)
    projected=autoencoder.transform(dataset)
    print("Save to file") 
    utils.save_object(out_path,projected)
    if(csv):
        print("Save to csv file") 
        csv_path=out_path.replace(".obj",".csv")
        utils.to_csv_file(csv_path,projected)

if __name__ == "__main__":
    path="/home/user/df/"
    in_path=path+"input/frames/"
    exp_path=path+"exp2/"
    conf=DfConf(exp_path)
    if(len(sys.argv)>0):
        save_reduction(in_path,conf.vectors,conf.nn)
    else:
        train_autoencoder(in_path,conf.nn,40)
