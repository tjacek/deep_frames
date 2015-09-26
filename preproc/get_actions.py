import os
from get_frames import get_all_files

def get_actions(in_path,out_path): 
    all_files = get_all_files(in_path)
    in_paths=map(lambda s:in_path+s,all_files)
    actions=map(lambda s:s.replace(".nonzero",""),all_files)
    for i,action in enumerate(actions):
        full_in_path=in_paths[i]
        full_out_path=out_path+action
        make_dir(full_out_path)
        decompose_action(full_in_path,full_out_path,i)

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path) 

def decompose_action(in_file,out_file,index):
    out_file+="/frame_"+str(index)+"_"
    print(out_file)
    cmd="th decompose_action.lua " + in_file+" "+out_file
    os.system(cmd)

if __name__ == "__main__":
    in_path="/home/user/Desktop/nonzero_data/"
    out_path="/home/user/df/actions/" 
    get_actions(in_path,out_path)
