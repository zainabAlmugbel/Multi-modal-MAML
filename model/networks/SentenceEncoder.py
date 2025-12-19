from sentence_transformers import SentenceTransformer#, models
import torch
from torch import Tensor
import torch.nn as nn
import pdb

class SentenceEncoder(nn.Module):
    def __init__(self, emb_size=640, use_cuda=False, sentence_no=15): #75 total
        super().__init__()
        #self.device = 'cuda' if use_cuda else 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'    
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")#SentenceTransformer('distilbert-base-uncased', device=device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embedding_dimension = 384#768 # one sentence length
        self.out_features = emb_size
        self.in_size=sentence_no*self.embedding_dimension
        self.out_size=sentence_no*self.out_features
        self.fc = nn.Linear(self.embedding_dimension, self.out_features)
        self.merge_sentnces_fc = nn.Conv1d(sentence_no, 1, 1)

    def forward(self, x, fusion_method):
        #print("sentence: ",x)
        
        num_instance, Sent_emb_size = x.shape
        #print("SE :",num_instance, Sent_emb_size) #x: (75,384) for 1 image
        #pdb.set_trace()
        self.fc = self.fc.to(self.device)

        self.merge_sentences_fc = self.merge_sentnces_fc.to(self.device)

        #x = torch.reshape(x, (Sent_emb_size, num_instance))
        #print("x: ",x.shape) # (384,75)
        x_att = self.fc(x)#.clone().detach())#torch.tensor(x))#.clone().detach().requires_grad_(True)) # in (1 * 384) out (284*640) = final 1*640 
        #print("x_att:", x.shape, x_att)
        #print("x_att: ",x_att.shape) #=> (15,640)
        x = self.merge_sentences_fc(x_att).squeeze(0) # 640
        if fusion_method == 'attention':
            return x, x_att
        else:
            return x