import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")
import scipy.misc as image
import numpy as np
from sklearn import preprocessing

def normalize_images(in_path,frame_path,action_path):
    actions=utils.get_all_dirs(in_path)
    actions=utils.append_path(in_path,actions)
    images,image_names,dim=get_raw_images(actions)
    images=utils.normalize(images)
    images=map(lambda x:x.reshape(dim),images)
    save_frames(images,image_names,frame_path,dim)    
    save_actions(action_path,images,image_names,actions)

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
        image.imsave(full_path,img)

def save_actions(out_path,images,image_names,actions):
    for action_path in actions:
        action_name=utils.get_filename(action_path)
        dir_path=out_path+action_name+"/"
        utils.make_dir(dir_path)
        print(dir_path)
        frame_names=utils.get_all_files(action_path)
        for frame_name in frame_names:
            print(frame_name)
            index=image_names.index(frame_name)
            frame=images[index]
            full_path=dir_path+frame_name
            image.imsave(full_path,frame)
       
if __name__ == "__main__":
   in_action="/home/user/df/input/raw_actions/"
   frame_path="/home/user/df/input/frames/"
   out_action="/home/user/df/input/actions/"
   normalize_images(in_action,frame_path,out_action)
