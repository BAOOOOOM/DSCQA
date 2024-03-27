# coding: UTF-8
import time, os
import numpy as np
import random
from code4class_embedding.bert import Config
import argparse
from code4class_embedding.utils import build_dataset, build_iterator, get_time_dif, load_dataset, gettoken
from torch.utils.data import TensorDataset, DataLoader
import torch
import json

import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Salient triple classification')
parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.",)
parser.add_argument("--test_batch", default=200, type=int, help="Test every X updates steps.")

parser.add_argument("--data_dir", default="code4class_embedding", type=str, help="The task data directory.")
parser.add_argument("--model_dir", default="bert-base-uncased", type=str, help="The directory of pretrained models")
parser.add_argument("--output_dir", default='output/save_dict/', type=str, help="The path of result data and models to be saved.")
# models param
parser.add_argument("--max_length", default=64, type=int, help="the max length of sentence.")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--dropout", default=0.1, type=float, help="Drop out rate")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--hidden_size', type=int, default=1024,  help="hidden_size")

args = parser.parse_args()

config = Config(args)


torch.backends.cudnn.deterministic = True

model = SentenceTransformer('sentence-transformers/sentence-t5-large',device=config.device).to(config.device)
#Enhance the differentiation of semantic representations
#model.load_state_dict(torch.load('model/finetuned_sentence_t5_csqa.ckpt'))
print("The trained model was successfully loaded")

def get_formal_embedding(all_embedding,need_att=True,need_token=True,need_sen=True,is_only_question=False):
    attention_mask_tensor=None
    token_embeddings_tensor=None
    sentence_embedding_tensor=None
    if is_only_question==True:
        #print(all_embedding.keys())
        if need_att==True:
            attention_mask_tensor=all_embedding['attention_mask']
        if need_token==True:
            token_embeddings_tensor=all_embedding['token_embeddings']
        if need_sen==True:
            sentence_embedding_tensor=all_embedding['sentence_embedding']
    else:
        for i in range(len(all_embedding)):
            if need_att==True:
                #print(all_embedding.keys())
                now_attention_mask=all_embedding[i]['attention_mask']
                now_attention_mask=now_attention_mask.expand(1,now_attention_mask.size()[0])
                if attention_mask_tensor==None:
                    attention_mask_tensor=now_attention_mask
                else:
                    attention_mask_tensor=torch.cat((attention_mask_tensor,now_attention_mask),dim=0)

            if need_token==True:
                now_token_embeddings=all_embedding[i]['token_embeddings']
                now_token_embeddings=now_token_embeddings.expand(1,now_token_embeddings.size()[0],now_token_embeddings.size()[1])
                if token_embeddings_tensor==None:
                    token_embeddings_tensor=now_token_embeddings
                else:
                    token_embeddings_tensor=torch.cat((token_embeddings_tensor,now_token_embeddings),dim=0)

            if need_sen==True:
                now_sentence_embedding=all_embedding[i]['sentence_embedding']
                now_sentence_embedding=now_sentence_embedding.expand(1,now_sentence_embedding.size()[0])
                if sentence_embedding_tensor==None:
                    sentence_embedding_tensor=now_sentence_embedding
                else:
                    sentence_embedding_tensor=torch.cat((sentence_embedding_tensor,now_sentence_embedding),dim=0)

    return attention_mask_tensor,token_embeddings_tensor,sentence_embedding_tensor

def get_class_embedding(config, model, data_iter,device,output_all_lab_emb=False):
    #print("2222222222")
    start_time = time.time()
    config.device=device
    tmp_ner_label = []
    ner_label_num = {}
    ner_label_embedding = {}
    ner_label_dict = ["comparison","negation","causality","capable","plausibility","temporal","meronymy","unknown"]
    all_label_embedding=[]
    for i in range(len(ner_label_dict)):
        ner_label_num[ner_label_dict[i]] = 0
        ner_label_embedding[ner_label_dict[i]] = []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            #print(i)
            model.eval()
            sent, _, labels = batches
            all_emb=model.encode(sent,batch_size=config.batch_size,output_value=None,device=device)
            a_emb,token_emb,sen_emb=get_formal_embedding(all_emb,need_att=False,need_token=False)
            #print(cls)
            #print(cls.size())
            #print("ok")
            
            for c, l in zip(sen_emb, labels):
                c=c.view(1,768)
                if all_label_embedding==[]:
                    all_label_embedding=c
                else:
                    all_label_embedding=torch.cat((all_label_embedding,c),dim=0)
                #print(c.size())
                for i in range(len(ner_label_dict)):
                    if ner_label_dict[i] in l:
                        #print("1111")
                        if len(ner_label_embedding[ner_label_dict[i]]) == 0:
                            ner_label_embedding[ner_label_dict[i]] = c
                        else:
                            ner_label_embedding[ner_label_dict[i]]=torch.cat((ner_label_embedding[ner_label_dict[i]],c),dim=0)
                        ner_label_num[ner_label_dict[i]] += 1
    
    for i in range(len(ner_label_dict)):
        if len(ner_label_embedding[ner_label_dict[i]]) != 0:
            #print(ner_label_embedding[ner_label_dict[i]].size())
            ner_label_embedding[ner_label_dict[i]] = torch.mean(ner_label_embedding[ner_label_dict[i]], dim=0)
    '''
    for i in range(len(ner_label_dict)):
        print(ner_label_dict[i])
        if len(ner_label_embedding[ner_label_dict[i]]) != 0:
            ner_label_embedding[ner_label_dict[i]] = model.dropout(ner_label_embedding[ner_label_dict[i]])
    '''
    return ner_label_embedding

def get_class_embedding_entry(device):
    start_time = time.time()
    config.device=device
    print("Loading data...")
    train_data_all = load_dataset(config.train_path, config)
    random.shuffle(train_data_all)
    train_data = train_data_all
    data_iter = DataLoader(
        train_data,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    #model = Model(config).to(config.device)

    ner_label_embedding=get_class_embedding(config, model, data_iter,device)
    #print(ner_label_embedding)
    return ner_label_embedding


def get_question_embedding(sent,device,batch_size,is_only_question=False):
    config.device=device
    #if is_only_question==True:
    #    print("true")
    with torch.no_grad():
        all_emb=model.encode(sent,batch_size=batch_size,output_value=None,device=device)
        a_emb,token_emb,sen_emb=get_formal_embedding(all_emb,is_only_question=is_only_question)
    return a_emb,sen_emb,token_emb

def get_question_embedding_list(device):
    config.device=device
    question_embedding_dict={}
    question_seq_embedding_dict={}
    train_data = load_dataset(config.train_path, config)
    dev_data = load_dataset(config.test_path, config)
    train_iter = DataLoader(train_data, shuffle=False, batch_size=config.batch_size,
                          num_workers=config.num_workers, drop_last=False)
    dev_iter = DataLoader(dev_data, shuffle=False, batch_size=config.batch_size,
                          num_workers=config.num_workers, drop_last=False)
    #model = Model(config).to(config.device)
    with torch.no_grad():
        for i, batches in enumerate(train_iter):
            model.eval()
            sent, id, labels = batches
            all_emb=model.encode(sent,output_value=None,device=device)
            a_emb,token_emb,sen_emb=get_formal_embedding(all_emb)
            for c, s, i in zip(sen_emb, token_emb, id):
                question_embedding_dict.update({i:c})
                question_seq_embedding_dict.update({i:s})

        for i, batches in enumerate(dev_iter):
            model.eval()
            sent, id, labels = batches
            all_emb=model.encode(sent,output_value=None,device=device)
            a_emb,token_emb,sen_emb=get_formal_embedding(all_emb)
            for c, s, i in zip(sen_emb, token_emb, id):
                question_embedding_dict.update({i:c})
                question_seq_embedding_dict.update({i:s})
    
    return question_embedding_dict,question_seq_embedding_dict

if __name__ == '__main__':

    get_class_embedding_entry()

