import utils

class Action(object):
    def __init__(self,category,person,frames):
        self.category=category
        self.person=person
	self.frames=frames

def read_action(action_path):
    action_name=get_action_name(action_path)
    category=get_category(action_name)
    person=get_person(action_name)
    all_files=utils.get_all_files(action_path)
    all_files=utils.append_path(action_path+"/",all_files)
    frames=utils.read_images(all_files)
    return Action(category,person,frames)

def get_action_name(action_path):
    return action_path.split("/")[-1]

def get_category(action_name):
    raw_cat=action_name.split("_")[0]
    cat=raw_cat.replace("a","")
    return int(cat)

def get_person(action_name):
    raw_person=action_name.split("_")[1]
    person=raw_person.replace("s","")
    return int(person)

if __name__ == "__main__":
   action_path="/home/user/df/actions/a01_s01_e01_sdepth"
   read_action(action_path)
