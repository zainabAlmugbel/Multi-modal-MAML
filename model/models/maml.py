# this maml is for final fc tensor of size 1280
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from model.networks.SentenceEncoder import SentenceEncoder
import pdb

from model.models.attention import CrossModalAttention2D
from model.models.sentence_transformer_attention import TextImageAttentionWithSentenceTransformer
from model.models.SelfAttentionForTwoTexts import SelfAttentionForTwoTexts
from torchmultimodal.models.flava.transformer import TransformerEncoder



def update_params(loss, params, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())

    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    
   
    updated_params = OrderedDict()
   
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad


    return updated_params

def inner_train_step(model, support_imgs, support_kg, query_kg, args,FLAVA_Encoder=None, cross_attention=None, con_dim=None, sentenceTran_cross_attention=None, img_to_seq=None, texts_model=None, text_seq=None, fusion_projection=None, text_projection=None):
    """ Inner training step procedure. """
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())

    label = torch.arange(args.way).repeat(args.shot)
    device = None
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
        device = "cuda"
    else:
        label = label.type(torch.LongTensor)  
        device = "cpu"       
    
        
    shots = support_imgs.shape[0]
    for _ in range(args.inner_iters):
        
        support_image_feature= model(support_imgs, updated_params, embedding = True)##
        support_kg_feature = support_kg
        query_kg_feature= query_kg
        size=support_image_feature.size()[1]
       
        if args.exp_name in ['Exp4','Exp5','Exp6', 'Exp7','Exp8','Exp9']:
            support_feature= torch.zeros((shots , 2*size),device=device)
        else:
            support_feature= torch.zeros((shots , size),device=device)
        


        if args.exp_name == 'Exp1':
            support_feature_i =  torch.cat((support_image_feature, support_kg_feature), 1)
            support_feature_i = fusion_projection(support_feature_i) #support_feature_i#

        elif args.exp_name == 'Exp2':
            support_feature_i =  torch.cat((support_image_feature, support_kg_feature), 1)
            support_feature_i = fusion_projection(support_feature_i)
        
        elif args.exp_name == 'Exp3':
            support_feature_i =  torch.cat((support_image_feature, support_kg_feature), 1)

        elif args.exp_name == 'Exp4':
            support_kg_feature = text_projection(support_kg_feature)
            support_feature_i =  torch.cat((support_image_feature, support_kg_feature), 1)
        
        elif args.exp_name == 'Exp5':
            support_kg_feature = text_projection(support_kg_feature)
            support_feature_i =  torch.cat((support_image_feature, support_kg_feature), 1)

        elif args.exp_name == 'Exp6':  
            processed_text1, processed_text2, attention_weights = texts_model(support_kg_feature, query_kg_feature)
            processed_text1= text_seq(processed_text1)
            support_feature_i = torch.cat((support_image_feature,processed_text1), 1)
            support_feature_i = support_feature_i

        elif args.exp_name == 'Exp7':  
            att_img_feature,_ = cross_attention(support_kg_feature,support_image_feature) 
            support_feature_i = torch.cat((support_image_feature,att_img_feature), 1)

        elif args.exp_name == 'Exp8':
            #attended_text, attended_imgs, attn_weights = con_attention_model(query_kg_feature, support_image_feature)
            batch_size = support_image_feature.size(0)
            img_seq_features = img_to_seq(support_image_feature)  # [batch_size, 640*8]
            img_seq_features = img_seq_features.view(batch_size, 8*8, 640//8)  # [batch_size, 8, 640]
            support_kg_feature= text_seq(support_kg_feature)

            outputs = sentenceTran_cross_attention(support_kg_feature, support_image_feature)#img_seq_features)

            support_feature_i = torch.cat((outputs["attended_text"],outputs["img_features"]), 1)
            support_feature_i = con_dim(support_feature_i)
            support_feature_i = torch.cat((support_feature_i,support_kg_feature), 1)

        elif args.exp_name == 'Exp9':
            #concat
            batch_size = support_image_feature.size(0)
            img_seq_features = img_to_seq(support_image_feature) 
            img_seq_features = img_seq_features.view(batch_size, 8*8, 640//8) 
            
            support_kg_feature = text_seq(support_kg_feature)
            txt_seq_features = img_to_seq(support_kg_feature)
            txt_seq_features = txt_seq_features.view(batch_size, 8*8, 640//8)  
        
            con_support_feature_i = torch.cat((img_seq_features,txt_seq_features), 1)
            output= FLAVA_Encoder(con_support_feature_i)[0]

            features= output
            features=features.view(features.size(0), -1)# [4,10240]
            support_feature_i = features 
            
            support_feature_i = features#sqkg_projection(features)
        
            #features=features.view(features.size(0), -1) 
        support_feature = support_feature_i



        support_feature= F.linear(support_feature, weight=updated_params['fc.weight'], bias=updated_params['fc.bias']) #320->4
 
        loss = F.cross_entropy(support_feature, label)


        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)
    return updated_params

class MAML(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12_maml import ResNetMAML
            self.encoder = ResNetMAML(dropblock_size=args.dropblock_size) 
        else:
            raise ValueError('')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
         # 2*350 for tf_transformer combiniation
        self.args = args

        #initialization is required for all layer inside if stats to avoid error
        self.encoder.fc =None
        self.fusion_projection= None
        self.text_projection=None
        self.texts_model=None
        self.text_seq=None
        self.cross_model=None
        self.cross_modal_attention=None
        self.con_reduce=None
        self.img_to_seq=None
        self.FLAVA_Encoder=None
        if args.exp_name == 'Exp1':
                
            self.voc_size = args.voc_size    #200
            self.encoder.fc = nn.Linear(hdim , args.way) #hdim emb_size + 
            self.encoder.fc = self.encoder.fc.to(self.device)            
            self.fusion_projection = nn.Sequential(
                    nn.Linear(self.voc_size + hdim , hdim),
                    nn.LayerNorm(hdim),
                    nn.Dropout(0.1),
                    nn.ReLU()
                ) 

        elif args.exp_name == 'Exp2':
            self.voc_size = args.voc_size  # this and next exp 350
            self.encoder.fc = nn.Linear(hdim , args.way)
            self.encoder.fc = self.encoder.fc.to(self.device)            
            self.fusion_projection = nn.Linear(self.voc_size + hdim , hdim)

        elif args.exp_name == 'Exp3':
            self.voc_size = args.voc_size
            self.encoder.fc = nn.Linear(self.voc_size + hdim , args.way)  
            self.encoder.fc = self.encoder.fc.to(self.device)            

        elif args.exp_name == 'Exp4':
            self.voc_size = args.voc_size
            self.encoder.fc = nn.Linear(hdim  + hdim, args.way) #hdim
            self.text_projection = nn.Sequential(nn.Linear(self.voc_size, hdim),nn.ReLU())
            #self.fusion_projection = nn.Linear(hdim + hdim , hdim)
            self.encoder.fc = self.encoder.fc.to(self.device)            


        elif args.exp_name == 'Exp5':
            self.voc_size = args.voc_size
            self.encoder.fc = nn.Linear(hdim  + hdim, args.way) #hdim
            self.text_projection = nn.Linear(self.voc_size, hdim)
            #self.fusion_projection = nn.Linear(hdim + hdim , hdim)
            self.encoder.fc = self.encoder.fc.to(self.device)            

        elif args.exp_name == 'Exp6':
            self.voc_size = args.voc_size
            self.texts_model = SelfAttentionForTwoTexts(self.voc_size, hdim)
            self.encoder.fc = nn.Linear(hdim  + hdim ,args.way)
            self.encoder.fc = self.encoder.fc.to(self.device)            

            self.text_seq = nn.Sequential(nn.Linear(self.voc_size,hdim),
                nn.LayerNorm(hdim ),
                nn.ReLU(),
                nn.Dropout(0.1),                             
                nn.Linear(hdim , hdim ))
            
        elif args.exp_name == 'Exp7':
            self.voc_size = args.voc_size
            self.cross_model= CrossModalAttention2D(self.voc_size, hdim, hdim)           
            self.encoder.fc = nn.Linear(hdim  + hdim ,args.way)
            self.encoder.fc = self.encoder.fc.to(self.device) 


        elif args.exp_name == 'Exp8':
            self.voc_size = args.voc_size
            self.cross_modal_attention = TextImageAttentionWithSentenceTransformer(
                text_dim = hdim, #self.voc_size,  # MiniLM-L6-v2 has 384 dimensions
                img_dim = hdim,  # ResNet12 640 dimensions 8 regions and 8 heads //8
                hidden_dim = hdim,
                num_classes = args.way
            )
            self.encoder.fc = nn.Linear(hdim  + hdim ,args.way)
            self.con_reduce = nn.Linear(2*hdim, hdim)
            self.img_to_seq = nn.Sequential(
                nn.Linear(hdim, hdim * 8),  # Convert to sequence
                nn.ReLU()
            )  
            self.text_seq = nn.Sequential(nn.Linear(self.voc_size,hdim),
                nn.LayerNorm(hdim ),
                nn.ReLU(),
                nn.Dropout(0.1),                             
                nn.Linear(hdim , hdim ))
            self.encoder.fc = self.encoder.fc.to(self.device)            
            self.con_reduce = self.con_reduce.to(self.device)


        elif args.exp_name == 'Exp9':
            self.voc_size = args.voc_size
            self.encoder.fc = nn.Linear(10240 ,args.way)
            self.encoder.fc = self.encoder.fc.to(self.device)                    
            self.text_seq = nn.Sequential(nn.Linear(self.voc_size,hdim),
                nn.LayerNorm(hdim ),
                nn.ReLU(),
                nn.Dropout(0.1),                             
                nn.Linear(hdim , hdim ))
            
            # Convert 2D image features to sequence form
            self.img_to_seq = nn.Sequential(
                nn.Linear(hdim, hdim * 8),  # Convert to sequence
                nn.ReLU()
            )            
            self.FLAVA_Encoder= TransformerEncoder(n_layer = 5, d_model= 640//8, n_head =8, dim_feedforward=2*hdim)# mulitmodal encoder include the attention


            

    



        

    
    def forward(self, support_imgs, query_imgs, support_txt, query_txt):
        
        # update with gradient descent
        updated_params = inner_train_step(self.encoder, support_imgs, support_txt, query_txt, self.args, self.FLAVA_Encoder, self.cross_model , self.con_reduce, self.cross_modal_attention, self.img_to_seq,self.texts_model, self.text_seq, self.fusion_projection, self.text_projection)
        
        # Query part
        qshots = query_imgs.shape[0]
        query_image_feature= self.encoder(query_imgs,updated_params, embedding=True)
        query_kg_feature =  query_txt
        support_kg_feature = support_txt

        size=query_image_feature.size()[1]

        if self.args.exp_name in ['Exp4','Exp5','Exp6', 'Exp7','Exp8','Exp9']:
            query_feature= torch.zeros((qshots , 2*size),device=self.device)
        else:
            query_feature= torch.zeros((qshots , size),device=self.device)


        if self.args.exp_name == 'Exp1':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = query_kg_feature
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1)
            query_feature_i = self.fusion_projection(query_feature_i)

        elif self.args.exp_name == 'Exp2':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = query_kg_feature
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1)
            query_feature_i = self.fusion_projection(query_feature_i)
        
        elif self.args.exp_name == 'Exp3':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = query_kg_feature
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1)

        elif self.args.exp_name == 'Exp4':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = self.text_projection(query_kg_feature)
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1)
            

        elif self.args.exp_name == 'Exp5':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = self.text_projection(query_kg_feature)
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1)
            

        elif self.args.exp_name == 'Exp6':  #640 
            processed_text1, processed_text2, attention_weights = self.texts_model(support_kg_feature, query_kg_feature)
            processed_text2=self.text_seq(processed_text2)               
            query_feature_i = torch.cat((query_image_feature,processed_text2), 1)
            query_feature_i = query_feature_i 

        elif self.args.exp_name == 'Exp7':   #640
            att_img_feature,_ = self.cross_model(query_kg_feature,query_image_feature)  
            query_feature_i = torch.cat((query_image_feature,att_img_feature), 1)

        elif self.args.exp_name == 'Exp8':

            batch_size = query_image_feature.size(0)
            img_seq_features = self.img_to_seq(query_image_feature)  # [batch_size, 640*8]
            img_seq_features = img_seq_features.view(batch_size, 8*8, 640//8)  # [batch_size, 8, 640]
            query_kg_feature= self.text_seq(query_kg_feature)
            # Cross-modal attention
            outputs = self.cross_modal_attention(query_kg_feature, img_seq_features)
            query_feature_i = torch.cat((outputs["attended_text"],outputs["img_features"]), 1)
            query_feature_i = self.con_reduce(query_feature_i)
            query_feature_i = torch.cat((query_feature_i,query_kg_feature), 1) 

        elif self.args.exp_name == 'Exp9':
            batch_size = query_image_feature.size(0)
            img_seq_features = self.img_to_seq(query_image_feature)  # [batch_size, 640*8]
            img_seq_features = img_seq_features.view(batch_size, 8*8, 640//8)  # [batch_size, 8, 640]
            
            query_kg_feature = self.text_seq(query_kg_feature)
            txt_seq_features = self.img_to_seq(query_kg_feature)
            txt_seq_features = txt_seq_features.view(batch_size, 8*8, 640//8)  # [batch_size, 8, 640]
            con_query_feature_i = torch.cat((img_seq_features,txt_seq_features), 1) #[4, 128, 80]
            output= self.FLAVA_Encoder(con_query_feature_i)[0]
            features= output
            features=features.view(features.size(0), -1)
            query_feature_i = features


            #= output
            features=features.view(features.size(0), -1)
            query_feature_i = features
            
        query_feature= query_feature_i


        #if self.args.fusion_method == 'attention':    
        logitis =   F.linear(query_feature, weight=updated_params['fc.weight'], bias=updated_params['fc.bias']) / self.args.temperature 

        return logitis
    
    def forward_eval(self, support_imgs, query_imgs, support_txt, query_txt):
        # update with gradient descent
        self.train()
        updated_params = inner_train_step(self.encoder, support_imgs, support_txt, query_txt, self.args,self.FLAVA_Encoder,  self.cross_model, self.con_reduce, self.cross_modal_attention, self.img_to_seq,self.texts_model, self.text_seq, self.fusion_projection,self.text_projection)

        
        # Query part
        qshots = query_imgs.shape[0]
        
        # iterate over every img and kg
        query_image_feature= self.encoder(query_imgs,updated_params, embedding=True)
        query_kg_feature= query_txt
        support_kg_feature = support_txt
        size=query_image_feature.size()[1]
        
        if self.args.exp_name in ['Exp4','Exp5','Exp6', 'Exp7','Exp8','Exp9']:
            query_feature= torch.zeros((qshots , 2*size),device=self.device)
        else:
            query_feature= torch.zeros((qshots , size),device=self.device)

        
        if self.args.exp_name == 'Exp1':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = query_kg_feature
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1)
            query_feature_i = self.fusion_projection(query_feature_i)

        elif self.args.exp_name == 'Exp2':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = query_kg_feature
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1) 
            query_feature_i = self.fusion_projection(query_feature_i)
        
        elif self.args.exp_name == 'Exp3':
            query_image_feature_i = query_image_feature
            query_kg_feature_i = query_kg_feature
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1) 

        elif self.args.exp_name == 'Exp4':
            query_image_feature_i = query_image_feature
            query_kg_feature_i =  self.text_projection(query_kg_feature)
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1) 

        elif self.args.exp_name == 'Exp5':
            query_image_feature_i = query_image_feature
            query_kg_feature_i =  self.text_projection(query_kg_feature)
            query_feature_i = torch.cat((query_image_feature_i, query_kg_feature_i), 1) 
            
        elif self.args.exp_name == 'Exp6':  #640 
            processed_text1, processed_text2, attention_weights = self.texts_model(support_kg_feature, query_kg_feature)
            processed_text2 = self.text_seq(processed_text2)
            query_feature_i = torch.cat((query_image_feature,processed_text2), 1)
            query_feature_i = query_feature_i

        elif  self.args.exp_name == 'Exp7':   #640
            att_img_feature,_ = self.cross_model(query_kg_feature,query_image_feature)    
            query_feature_i = torch.cat((query_image_feature,att_img_feature), 1)

        elif  self.args.exp_name == 'Exp8': 
            batch_size = query_image_feature.size(0)
            img_seq_features = self.img_to_seq(query_image_feature)  # [batch_size, 640*8]
            img_seq_features = img_seq_features.view(batch_size, 8*8, 640//8)  # [batch_size, 8, 640]
            query_kg_feature= self.text_seq(query_kg_feature)
            # Cross-modal attention
            outputs = self.cross_modal_attention(query_kg_feature, img_seq_features)##needs to update the initalization to make it work
            
            query_feature_i = torch.cat((outputs["attended_text"],outputs["img_features"]), 1)
            query_feature_i = self.con_reduce(query_feature_i)
            query_feature_i = torch.cat((query_feature_i,query_kg_feature), 1)

        elif self.args.exp_name == 'Exp9':
            batch_size = query_image_feature.size(0)
            img_seq_features = self.img_to_seq(query_image_feature)  
            img_seq_features = img_seq_features.view(batch_size, 8*8, 640//8)  

            query_kg_feature = self.text_seq(query_kg_feature)
            txt_seq_features = self.img_to_seq(query_kg_feature)   # convert text to sequence
            txt_seq_features = txt_seq_features.view(batch_size, 8*8, 640//8) 
            con_query_feature_i = torch.cat((img_seq_features,txt_seq_features), 1) 
            features= self.FLAVA_Encoder(con_query_feature_i)[0]
            features=features.view(features.size(0), -1)
            
            query_feature_i = features


            #output
            features= query_feature_i
            features=features.view(features.size(0), -1)
            query_feature_i = features #changed the torward dim of FLAVA_Encoder 640 -> 1280

        query_feature= query_feature_i

        self.eval()
        with torch.no_grad():
            logitis =   F.linear(query_feature, weight=updated_params['fc.weight'], bias=updated_params['fc.bias']) / self.args.temperature 
                    
        return logitis
    
 