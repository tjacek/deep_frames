import utils
from actions import read_action
from clustering import Clusters
from reduction.autoencoder import AutoEncoder,AutoEncoderReduction

def get_deep_frames(action_path,reduction,cls):
    action=read_action(action_path)
    print(action.frames[0].shape)
    images=reduction.transform(action.frames)
    assigment=cls.find_clusters(images)
    action.set_seq(assigment)
    print(action)
    return action

if __name__ == "__main__":
   action_path="/home/user/df/actions/a01_s01_e01_sdepth"
   nn_path="/home/user/df/ae"
   cls_path="/home/user/df/clust.obj"
   reduction=AutoEncoderReduction(nn_path)
   cls=Clusters(cls_path)
   get_deep_frames(action_path,reduction,cls)
