#!/usr/bin/python
import utils
from actions import read_action
from clustering import Clusters
from reduction.autoencoder import AutoEncoder,AutoEncoderReduction

def create_seq_dataset(action_dir,nn_path,cls_path,out_path):
    action_files=utils.get_all_dirs(action_dir)
    #print(action_files)
    action_files=utils.append_path(action_dir,action_files)
    reduction=AutoEncoderReduction(nn_path)
    cls=Clusters(cls_path)
    def curried(action_path):
        return get_deep_frames(action_path,reduction,cls)
    actions=map(curried,action_files)
    seq_data=utils.array_to_txt(actions)
    utils.save_string(out_path,seq_data)
      
def get_deep_frames(action_path,reduction,cls):
    print(action_path)
    action=read_action(action_path)
    images=reduction.transform(action.frames)
    assigment=cls.find_clusters(images)
    action.set_seq(assigment)
    print(action)
    return action

if __name__ == "__main__":
   action_path="/home/user/df/actions/"
   nn_path="/home/user/df/ae"
   cls_path="/home/user/df/clust.obj"
   out_path="/home/user/df/dataset.seq"
   create_seq_dataset(action_path,nn_path,cls_path,out_path)
