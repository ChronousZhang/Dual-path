#coding:utf-8
import numpy as np 
import gensim 
import os
import string,re
from tqdm import tqdm
from flickr30k import flickr30k
import shutil


#Delete (,.)
regex = re.compile('[%s]' % re.escape(string.punctuation))
def test_re(s): 
    return regex.sub('', s)

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary = True)
padding = [0]*300


#######################################################
print("#"*20)
base_path = '/data/reid/flickr30k/Dual-path/'


def parse_data(data, part):
    for index, (img_path, text) in enumerate(data.items()):
        new_text = []
        
        # save image to correct path
        if not os.path.exists(os.path.join(base_path, part)):
            os.mkdir(os.path.join(base_path, part))
        if not os.path.exists(os.path.join(base_path, part, str(index))):
            os.mkdir(os.path.join(base_path, part, str(index)))
        new_file_path = os.path.join(base_path, part, str(index), os.path.basename(img_path))
        shutil.copy(img_path, new_file_path)
        for sub_index, _ in enumerate(text):
            lis = test_re(_)    #delete ',.'
            content = lis.strip('\n').split(' ')
            feature_map = []
            num = np.random.randint(10)   ## Random start position
            feature_map.extend([padding]*num)   ####position shift 
            for word in content:
                #print(word)
                if word in model.vocab:
                    feature = model[word]
                    if len(feature_map)<56:
                        feature_map.append(feature)
                    else:
                        break      
                else:
                    continue
                  
            while len(feature_map)<56:
                feature_map.append(padding)
            new_text.append(feature_map)
            # save text2vector to correct path
            np.save(os.path.join(base_path, part, str(index), str(sub_index) +'.npy'), feature_map)
        # print(img_path, new_text)
        # data[img_path] = new_text
        # for a small dataset debuging
        #if index > 10:
        #    break

def word2vector():
    train_data, val_data, test_data = flickr30k()
    train_data = parse_data(train_data, 'train')
    val_data = parse_data(val_data, 'val')
    test_data = parse_data(test_data, 'test')

word2vector()
