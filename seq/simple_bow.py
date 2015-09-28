import seq
import numpy as np

n_cls=10

def to_bow(instances):
    return map(to_bow_vector,instances)

def to_bow_vector(instance):
    bow=np.zeros(n_cls)
    for index in instance.seq:
        bow[index]+=1.0
    bow/=sum(bow)
    return bow,instance.category

if __name__ == "__main__":
   path="/home/user/df/exp2/dataset.seq"
   instances=seq.parse_dataset(path)
   vectors=to_bow(instances)
   out_path="/home/user/df/exp2/dataset.lb"
   seq.to_labeled(out_path,vectors)
