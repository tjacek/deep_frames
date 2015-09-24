import pickle
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn import preprocessing

def get_all_files(path):
    all_files= [ f for f in listdir(path) if isfile(join(path,f)) ]
    all_files.sort()
    return all_files

def append_path(path,files):
    return map(lambda f: path+f,files)

def normalize(data):
    if(isinstance(data,list)):
        data=np.array(data)
    data=data.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data

def save_object(out_path,nn):
    file_object = open(out_path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

