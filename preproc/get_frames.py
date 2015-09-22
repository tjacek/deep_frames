import os
from os import listdir
from os.path import isfile, join
import subprocess

def get_frames(in_path,out_path): 
    all_files = get_all_files(in_path)
    for filename in all_files:
        in_file=in_path+filename
        out_file=out_path+filename
        transform_file(in_file,out_file)

def get_all_files(path):
    all_files= [ f for f in listdir(path) if isfile(join(path,f)) ]
    all_files.sort()
    return all_files

def transform_file(in_file,out_file):
    out_file=out_file.replace(".nonzero","")
    print(out_file)
    cmd="th decompose_action.lua " + in_file+" "+out_file
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    in_path="/home/user/Desktop/nonzero_data/"
    out_path="/home/user/df/frames/" 
    get_frames(in_path,out_path)
