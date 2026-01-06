import os.path as osp
import PIL
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import jpeg4py as jpeg
import h5py
import pdb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from model.dataloader.fastText import FastTextclass
from multiprocessing import Pool

import itertools
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/cub/CUB_200_2011/images'
SPLIT_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/cub/CUB_200_2011/full_split'
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
TEXT_PATH = '/mnt/scratch/users/zha503/Unicorn Folder/data/cub/CUB_200_2011/text_c10'


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

class CUB(Dataset):

    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data, self.txts, self.label = self.parse_csv(csv_path)
        self.num_class = len(set(self.label))


               
        self.args = args
        self.corpus = list(itertools.chain.from_iterable(self.txts))
        #print(len(self.corpus))693
        self.terms= None
        self.tfidf_matrix = None
        image_size = 32
        self.transform_aug, self.transform = get_transforms(image_size, args.backbone_class)
        self.text_size= args.voc_size

        if args.text_encoder == 'TF_IDF':
            self.tfidf_vectorizer = TfidfVectorizer(max_features=self.text_size, ngram_range=(1,2), use_idf=True, norm='l2')
            self.tfidf_vectorizer.fit(self.corpus)
        elif args.text_encoder == 'FastText':
            self.fastencoder= FastTextclass()
        elif args.text_encoder == 'Transformer':
            self.transformer_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        elif args.text_encoder == 'wTF_Transformer':
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, norm='l2')
            self.tfidf_vectorizer.fit_transform(self.corpus)
            self.voc_ind= self.tfidf_vectorizer.vocabulary_
            #self.tfidf_vectorizer.transform(self.corpus)
            #print(self.tfidf_vectorizer.fit_transform(self.corpus))
            #for term in self.tfidf_vectorizer.get_feature_names_out():
                #print(term)
            #print("**************")
           # self.tokenizer = AutoTokenizer.from_pretrained('all-MiniLM-L6-v2')
            self.transformer_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    def parse_csv(self, csv_path):
        data = []
        label = []
        stats= []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')

            path = osp.join(IMAGE_PATH, name)
            #foldname,imgname = name.split('/')
            text_path= osp.join(TEXT_PATH, name[:-3])
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            Stat_vec = [x for x in open(text_path+"txt", 'r').readlines()]
            with h5py.File(text_path+'h5', 'r') as file:
                a_group_key = list(file.keys())[0]
            # Getting the data
                txtdata = list(file[a_group_key])

            stats.append(Stat_vec)
            data.append(path)
            label.append(lb)

        return data, stats, label


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

        self.tfidf_matrix = self.tfidf_vectorizer.transform(descriptions)
        self.terms= self.tfidf_vectorizer.inverse_transform(self.tfidf_matrix)

        return self.tfidf_matrix
    

    def __getitem__(self, i):
        data, text_data, label = self.data[i], self.txts[i], self.label[i]

                #print(len(text_data) ) get 75 stats per sample correct
        if self.args.text_encoder == 'TF_IDF':
            self.extract_text_features(text_data)
            text_normalized=normalize_size_pad_truncate(self.tfidf_matrix.toarray().flatten(),self.text_size)
        elif self.args.text_encoder == 'FastText':
            text_normalized= normalize_size_pad_truncate(self.fastencoder.get_sentence_embedding( ' '.join(text_data)),self.text_size)
        elif self.args.text_encoder == 'Transformer':
            text_normalized= normalize_size_pad_truncate(self.transformer_encoder.encode(' '.join(text_data)),self.text_size)

        try:
            image = self.transform(Image.fromarray(jpeg.JPEG(data).decode()).convert('RGB'))
        except:
            image = self.transform(Image.open(data).convert('RGB'))
        
  
        return image, text_normalized, label
