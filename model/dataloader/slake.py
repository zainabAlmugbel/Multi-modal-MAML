import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import jpeg4py as jpeg
from sentence_transformers import models,SentenceTransformer
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from model.dataloader.fastText import FastTextclass
import pdb
import itertools
#from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.models.flava.model import flava_model

from model.bert_flava_text import flava_text_only_approach,TextFeatureExtractor

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
IMAGE_PATH1 = '/mnt/scratch/users/zha503/Unicorn Folder/data/slake/images/'
#IMAGE_PATH2 = osp.join(ROOT_PATH2, 'data/miniimagenetaux/images')
SPLIT_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/slake/split/'  #split for Random class distribution
#KG_File='/mnt/scratch/users/zha503/Unicorn Folder/data/slake/KG_Slake.txt'
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
split_map = {'train':IMAGE_PATH1, 'val':IMAGE_PATH1, 'test':IMAGE_PATH1}#, 'aux_val':IMAGE_PATH2, 'aux_test':IMAGE_PATH2}

def identity(x):
    return x

    
def get_transforms(size, backbone, s = 1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    if backbone == 'ConvNet':
        normalization = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                             np.array([0.229, 0.224, 0.225]))       
    elif backbone == 'Res12':
        normalization = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                             np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
    elif backbone == 'Res18' or backbone == 'Res50':
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    else:
        raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
    
    data_transforms_aug = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.ToTensor(),
                                              normalization])
    
    data_transforms = transforms.Compose([transforms.Resize(size + 8),
                                          transforms.CenterCrop(size),
                                          transforms.ToTensor(),
                                          normalization])
    
    return data_transforms_aug, data_transforms


def normalize_size_pad_truncate(text_feature, target_size=800): # number of words
    """Normalize text feature vector to a fixed size by padding or truncation."""
    current_size = len(text_feature)
    
    if current_size > target_size:
        # Truncate to target size
        return text_feature[:target_size]
    elif current_size < target_size:
        # Pad with zeros to target size
        return np.pad(text_feature, (0, target_size - current_size))
    else:
        # Already the right size
        return text_feature

class Slake(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')

        self.data, self.statement, self.label, self.masks, self.spatials = self.parse_csv(csv_path, setname)
        
        self.num_class = len(set(self.label))
        self.args = args
        self.corpus = list(itertools.chain.from_iterable(self.statement))
        #print(len(self.corpus))693
        self.terms= None
        self.tfidf_matrix = None

        image_size = 32
        self.text_size= 350
        self.transform_aug, self.transform = get_transforms(image_size, args.backbone_class)
        if args.text_encoder == 'TF_IDF':
            self.tfidf_vectorizer = TfidfVectorizer(max_features=self.text_size, ngram_range=(1,2), use_idf=True, norm='l2')
            self.tfidf_vectorizer.fit(self.corpus)
        elif args.text_encoder == 'FastText':
            self.fastencoder= FastTextclass()
        elif args.text_encoder == 'Transformer':
            #transformer= models.Transformer("all-MiniLM-L6-v2", max_seq_length=256)
            self.transformer_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        #elif args.text_encoder == 'flava_Transformer':
            #self.text_transform = BertTextTransform()
        #    print("###")
            
 

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = [] # data is image
        label = [] #class
        stats= [] # statements or KG
        masks= [] # the object
        spatial = [] # the location of the detected objects

        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            #foldname is needed for the mask and the spatial
            foldname,imgname = name.split('/')
            img_path = osp.join(split_map[setname], name)
            msk_path = osp.join(split_map[setname], "mask.png")
            sptil_path = osp.join(split_map[setname], "detection.json")
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(img_path)
            label.append(lb)
            masks.append(msk_path)
            spatial.append(sptil_path)

            Stat_words = [x.strip().replace("#",' ') for x in open(img_path[:-10]+'statment.txt', 'r').readlines()] # 10 is for the image name.jpg

            stats.append(Stat_words)

        return data, stats, label, masks, spatial
    
    def parseStt(self, path):
        with open(path) as f:
            lines = f.readlines()
        return lines[:20]    


       
    def __len__(self):
        return len(self.data)

 
    def extract_text_features(self, descriptions):
        """
        Extract TF-IDF features from text descriptions.
            
        Args:
            descriptions: List of text descriptions
                
        Returns:
            TF-IDF feature matrix
        """
        #self.corpus = descriptions
        self.tfidf_matrix = self.tfidf_vectorizer.transform(descriptions)
        self.terms= self.tfidf_vectorizer.inverse_transform(self.tfidf_matrix)
        #print("----------------") 
        #print(self.tfidf_matrix.shape) (8, 473)
        return self.tfidf_matrix
    
    def __getitem__(self, i):
        data, text_data, mask, spatial, label = self.data[i], self.statement[i], self.masks[i], self.spatials[i], self.label[i]
        #print(data)
        #print(text_data)
        
        encoded_list = []
        #print(len(text_data) ) get 75 stats per sample correct
        if self.args.text_encoder == 'TF_IDF':
            self.extract_text_features(text_data)
            text_normalized=normalize_size_pad_truncate(self.tfidf_matrix.toarray().flatten(),self.text_size)
        elif self.args.text_encoder == 'FastText':
            text_normalized= normalize_size_pad_truncate(self.fastencoder.get_sentence_embedding( ' '.join(text_data)),self.text_size)
        elif self.args.text_encoder == 'Transformer':
            text_normalized= normalize_size_pad_truncate(self.transformer_encoder.encode(' '.join(text_data)) ,self.text_size)
        elif self.args.text_encoder == 'flava_Transformer':
            #Textextractor= TextFeatureExtractor()
            for sentence in text_data:
                encoded_result = flava_text_only_approach(sentence)
                encoded_list.append(normalize_size_pad_truncate(encoded_result['text_features'],640))
            text_data = np.array(encoded_list)
            text_normalized = np.resize(text_data, (15,self.text_size))
            

        try:
            image = self.transform(Image.fromarray(jpeg.JPEG(data).decode()).convert('RGB'))
            #for stt in text_data:
            #    all_encode.append(self.text_encoder.encode(stt, convert_to_numpy=False))

        except:
            image = self.transform(Image.open(data).convert('RGB'))
            #kg_encode = self.text_encoder.encode(stat ,convert_to_numpy=False)
        
        #np.resize(self.tfidf_matrix.toarray().flatten(), (15,self.text_size))# i have variable no of sentences for that I need resize

        
        return image, text_normalized, label
