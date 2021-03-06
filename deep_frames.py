#!/usr/bin/python
import sys
import utils
from actions import read_action
from clustering import Clusters
from reduction.autoencoder import AutoEncoder,AutoEncoderReduction
from dim_reduction import DfConf

def create_seq_dataset(action_dir,conf):
    actions=get_actions(action_dir,conf.nn,conf.cls)
    seq_data=utils.array_to_txt(actions)
    utils.save_string(conf.seq,seq_data)

def get_actions(action_dir,nn_path,cls_path):
    action_files=utils.get_all_dirs(action_dir)
    action_files=utils.append_path(action_dir,action_files)
    reduction=AutoEncoderReduction(nn_path)
    cls=Clusters(cls_path)
    def curried(action_path):
        return get_deep_frames(action_path,reduction,cls)
    return map(curried,action_files)
  
def get_deep_frames(action_path,reduction,cls):
    print(action_path)
    action=read_action(action_path)
    images=reduction.transform(action.frames)
    assigment=cls.find_clusters(images)
    action.set_seq(assigment)
    print(action)
    return action

def get_all_clusters(action_path,conf,out_path,n_cls=10):
    actions=get_actions(action_path,conf.nn,conf.cls)
    symbols=actions[0].symbols
    for i in range(n_cls):
        cls_symbol=symbols[i]
        full_path=out_path+cls_symbol+"/"
        print(full_path)
        utils.make_dir(full_path)
        get_cluster(i,actions,full_path)

def get_cluster(index,actions,out_path):
    cls=map(lambda x:x.get_cluster(index),actions)
    cls=sum(cls,[])
    names=map(lambda i: out_path+"cls_"+str(i)+".jpg",range(len(cls)))
    for name,image in zip(names,cls):
        utils.save_image(name,image)

if __name__ == "__main__":
   action_path="/home/user/df/input/actions/"
   exp_path="/home/user/df/exp2/"
   conf=DfConf(exp_path)
   if(len(sys.argv)>0):
       create_seq_dataset(action_path,conf)
   else:
       show_path="/home/user/df/cls2/"
       get_all_clusters(action_path,conf,show_path)
