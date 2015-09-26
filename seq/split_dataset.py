import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")

class Instance(object):
    def __init__(self,category,person,seq):
        self.category=int(category)
        self.person=int(person)
        self.seq=seq

    def __str__(self):
        s=self.seq+'$'+str(self.category)
        return s+"\n"

def split_dataset(path):
    lines=utils.read_file(path)
    #lines=raw_txt.split("\n")
    #del lines[-1]
    instances=map(parse_line,lines)
    train=filter(odd_person,instances)
    test=filter(lambda x: not odd_person(x),instances)
    train_path=path.replace(".seq","_train.seq")
    test_path=path.replace(".seq","_test.seq")
    save_dataset(train_path,train)
    save_dataset(test_path,test)

def parse_line(line):
    action={}
    line=line.split('$')
    return Instance(line[1],line[2],line[0])

def odd_person(instance):
    return (instance.person % 2)==0

def save_dataset(path,instances):
    txt=utils.array_to_txt(instances)
    utils.save_string(path,txt)

if __name__ == "__main__":
   seq_path="/home/user/df/seq/dataset.seq"
   split_dataset(seq_path)
