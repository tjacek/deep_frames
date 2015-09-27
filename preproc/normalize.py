import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")
import scipy.misc as image
import numpy as np
from sklearn import preprocessing

def normalize_images(in_path,frame_path):
    actions=utils.get_all_dirs(in_path)
    actions=utils.append_path(in_path,actions)
    images,image_names,dim=get_raw_images(actions)
    images=utils.normalize(images)
    save_frames(images,image_names,frame_path,dim)    

def get_raw_images(actions):
    images=[]
    image_names=[]
    dim=None
    for action in actions:
        files=utils.get_all_files(action)
        for frame_name in files:
            print(action+"/"+frame_name)
            img=image.imread(action+"/"+frame_name)
            dim=img.shape
            img=utils.standarize_image(img)
            image_names.append(frame_name)
            images.append(img)
    images=np.array(images)
    return images,image_names,dim

def save_frames(images,image_names,frame_path,dim):
    for i,img in enumerate(images):
        full_path=frame_path+image_names[i]
        img=np.reshape(img,dim)
        print(img.shape)
        image.imsave(full_path,img)
       
if __name__ == "__main__":
   in_action="/home/user/df/input/raw_actions/"
   frame_path="/home/user/df/input/frames/"
   normalize_images(in_action,frame_path)
