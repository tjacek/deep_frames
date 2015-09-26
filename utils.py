import time
import pickle,re
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.misc as image
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

def save_string(path,string):
    file_str = open(path,'w')
    file_str.write(string)
    file_str.close()

def save_object(path,nn):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def to_csv_file(path,vectors):
    file_csv = open(path,'w')
    csv=""
    for i,instance in enumerate(vectors):
        print(i)
        str_v=vector_to_str(instance)
        csv+=str_v
    file_csv.write(csv)
    file_csv.close()

def vector_to_str(vector):
    v=str(vector).replace(" ]","")
    v=re.sub(r"\s+", ',', v)
    v=v.replace("[,","")
    v=re.sub("[\[\]]","",v)
    return v+"\n"

def read_array(path):
    arr=read_object(path)
    arr=map(lambda x:x.flatten(),arr)
    return np.array(arr)

def read_images(files):
    return map(lambda f:image.imread(f),files)

def flatten_images(images):
    return map(lambda img:np.reshape(img,(img.size)),images)

def show_images(imgs):
    for img in imgs:
        image.imshow(img)
        time.sleep(1)
