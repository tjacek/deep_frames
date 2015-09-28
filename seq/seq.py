import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")

symbols="ABCDEFGHIJKLMNOPRSTUVWXYZ"

class Instance(object):
    def __init__(self,category,person,seq):
        self.category=category
        self.person=person
        self.seq=seq

def parse_dataset(path):
    lines=utils.read_file(path)
    instances=map(parse_instance,lines)
    return instances

def get_frames(string):
    return map(get_index,string)

def get_index(symbol):
    return symbols.index(symbol)

def parse_instance(raw_instance):
    raw=raw_instance.split('$')
    seq=get_frames(raw[0])
    category=int(raw[1])
    person=int(raw[2])
    return Instance(category,person,seq)

def to_labeled(out_path,vectors):
    labeled=""
    for v in vectors:
        line=""
        for num in v[0]:
            value=round(num,3)
            line+=str(value)+","
        line+="#"+str(v[1])+"\n"
        labeled+=line
    utils.save_string(out_path,labeled)
