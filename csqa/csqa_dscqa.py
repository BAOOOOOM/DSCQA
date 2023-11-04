from openprompt import PromptModel
from openprompt.prompts import ManualTemplate
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt.plms.modeling_t5 import T5Config,T5ForConditionalGeneration
from transformers import AdamW
from torch import nn


import argparse
from concurrent.futures import process
import json
from tqdm import tqdm
import numpy as np
import os
import torch
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pdb
from typing import List
import random
from torch import cuda
import torch.nn.functional as F
import gc
#device=get_device()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") 

ANSWERS = ['yes', 'no']
Batch_Size=1


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default=None)
parser.add_argument('--input-path', type=str, default=None)
parser.add_argument('--test-path', type=str, default=None)

parser.add_argument('--average-loss', action='store_true')
parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob'])
parser.add_argument('--n', type=int, default=None)
parser.add_argument('--topk', type=int, default=3)


parser.add_argument("--plm_eval_mode", action="store_true")
args = parser.parse_args()
# args.output_path = f'data/{args.task}/inference/inference_{args.model_type.split("/")[-1]}.{args.input_path.split("/")[-1]}'

model_params = {
"MODEL": "t5-large",  # model_type: t5-base/t5-large
"TRAIN_BATCH_SIZE": 128,  # training batch size
"VALID_BATCH_SIZE": 128,  # validation batch size
"TRAIN_EPOCHS": 40,  # number of training epochs
"LEARNING_RATE_KG": 1e-5,  # learning rate
"LEARNING_RATE_INF": 1e-5,  # learning rate
"MAX_INPUT_KG_LENGTH": 64,  # max length of all text
"MAX_SOURCE_KG_LENGTH": 50,  # max length of source text
"MAX_TARGET_KG_LENGTH": 30,  # max length of target text
"MAX_SOURCE_INF_LENGTH": 64,  # max length of source text
"MAX_TARGET_INF_LENGTH": 10,  # max length of target text
"SEED": 42,  # set seed for reproducibility
}   

from code4class_embedding.sentence_t5_get_embedding import get_class_embedding_entry,get_question_embedding_list,get_question_embedding

class_embedding_dict=get_class_embedding_entry(device)
#print(class_embedding_dict)
print("get class embedding successful!")


# load generated knowledge from GPT-3
with open("new_csqa_8class_data/csqa_train.json") as f:
    ds = json.load(f)
    if args.n is not None:
        ds = ds[:args.n]

with open("new_csqa_8class_data/csqa_dev.json") as f:
    test_data = json.load(f)




def get_cat_class_emb(class_embedding_dict,batch_size):
    total_class_embedding="a"
    #print(type(class_embedding_dict))
    for key, value in class_embedding_dict.items():
        if total_class_embedding=="a":
            total_class_embedding=value.expand(1,value.size()[0])
            #print(value.size())
            #print(value)
        else:
            total_class_embedding=torch.cat((total_class_embedding,value.expand(1,value.size()[0])),dim=0)
    #print(total_class_embedding.size())
    #print(total_class_embedding)
    total_class_embedding=total_class_embedding.expand(batch_size,8,768) #torch.Size([8, 768])——>torch.Size([batch_size, 8, 768])
    total_class_embedding=total_class_embedding.permute(1,0,2)# torch.Size([8, batch_size, 768])
    return total_class_embedding

normal_cat_class_embedding=get_cat_class_emb(class_embedding_dict,Batch_Size)
#raise ValueError("test")
def id2seq_question_emb(sent,question_embedding_dict,question_seq_embedding_dict):
    question_seq_emb=[]
    question_emb=[]
    for id in sent:
        question_emb.append(question_embedding_dict[id])
        question_seq_emb.append(question_seq_embedding_dict[id])
    return question_emb,question_seq_emb





# tokenzier for encoding the text
tokenizer_inf = transformers.T5Tokenizer.from_pretrained(model_params["MODEL"])
model_inf = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])

#template_text = '{"placeholder":"text_a"} {"mask"}'
#template_text = '{"placeholder":"text_a"} {"mask"}'
#mytemplate = ManualTemplate(tokenizer=tokenizer_inf, text=template_text)
mytemplate = PrefixTuningTemplate(model=model_inf,  tokenizer=tokenizer_inf, text='{"placeholder":"text_a"} {"mask"} ', using_decoder_past_key_values=True,num_token=100,prefix_dropout=0.5)

import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None,temperature=1):
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores=scores/temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        #print(scores,p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,temperature=1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)] + [nn.Identity()])
        self.output_linear = nn.Identity()
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
        self.temperature=temperature

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout,temperature=self.temperature)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        #print(x.size())


        return self.output_linear(x)


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.prompt_model = PromptModel(plm=model_inf,template=mytemplate, freeze_plm=True, plm_eval_mode=args.plm_eval_mode)
        self.multihead_attn = MultiHeadAttention(12,768,temperature=0.2)
        torch.nn.init.eye_(self.multihead_attn.linear_layers[0].weight)
        torch.nn.init.zeros_(self.multihead_attn.linear_layers[0].bias)
        torch.nn.init.eye_(self.multihead_attn.linear_layers[1].weight)
        torch.nn.init.zeros_(self.multihead_attn.linear_layers[1].bias)
        torch.nn.init.eye_(self.multihead_attn.linear_layers[2].weight)
        torch.nn.init.zeros_(self.multihead_attn.linear_layers[2].bias)
        self.skill_emb_trans = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(512, 1024))
        #self.attention=nn.linear(q,k,v)
    def forward(self, inputs,sent):
        a_emb,question_embedding,question_seq_emb=get_question_embedding(sent,device,Batch_Size,is_only_question=True)
        question_embedding=question_embedding.expand(1,question_embedding.size()[0])
        cat_class_embedding=get_cat_class_emb(class_embedding_dict,question_embedding.size()[0])
        
        now_batch=question_embedding.size()[0]

        question_embedding=question_embedding.expand(1,question_embedding.size()[0],question_embedding.size()[1])#torch.Size([batch_size, 768])——>torch.Size([1, batch_size, 768])

        ###################get dynamic prefixes######################
        dynamic_class=self.multihead_attn(question_embedding,cat_class_embedding,cat_class_embedding)#torch.Size([1, batch_size, 768])+torch.Size([8, batch_size, 768])=torch.Size([1, batch_size, 768])
        #print(type(dynamic_class))
        dynamic_class=dynamic_class.permute(1,0,2)#torch.Size([1, batch_size, 768])——>torch.Size([batch_size, 1, 768])

        trans_cat_class_embedding=self.skill_emb_trans(cat_class_embedding)
        trans_cat_class_embedding=trans_cat_class_embedding.permute(1,0,2)

        logits = self.prompt_model(inputs,mulit_prefix=dynamic_class,cat_class_embedding=trans_cat_class_embedding)
        #raise ValueError("test")
        return logits
    

model_inf = Mymodel().to(device)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model_inf.multihead_attn.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model_inf.multihead_attn.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model_inf.skill_emb_trans.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model_inf.skill_emb_trans.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model_inf.prompt_model.plm.skill2question_multihead_attn.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model_inf.prompt_model.plm.skill2question_multihead_attn.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]

def checker(args, answer, pred):
    if args.task == 'numersense':
        if answer == pred:
            return 1
        if answer in ['no', 'zero'] and pred in ['no', 'zero']:
            return 1
        return 0
    return 1 if answer == pred else 0


def data_batch_inf(args, model_params, data, tokenizer_inf):
    source_inf, target_inf = [], []
    if 'cands' in data[0]:
        num_cands = len(data[0]['cands'])
    else:
        num_cands = len(ANSWERS)
    for item in data:
        query = item["query"]
        source_inf.append(query.replace('<mask>', '<extra_id_0>'))
        target_inf.append('<extra_id_0> %s <extra_id_1>' % item['answer'])


    source_inf = tokenizer_inf.batch_encode_plus(
        source_inf,
        max_length=model_params['MAX_SOURCE_INF_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    target_inf = tokenizer_inf.batch_encode_plus(
            target_inf,
            max_length=model_params['MAX_TARGET_INF_LENGTH'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


    return {
            "source_inf_ids": source_inf["input_ids"].to(dtype=torch.long),
            "source_inf_mask": source_inf["attention_mask"].to(dtype=torch.long),
            "target_inf_ids": target_inf["input_ids"].to(dtype=torch.long)
        }



def train_inf(tokenizer, model, data, optimizer,question_inputs):

    model.train()
    y = data["target_inf_ids"].to(device, dtype=torch.long)
    ids = data["source_inf_ids"].to(device, dtype=torch.long)
    ids[data["source_inf_ids"] == tokenizer.pad_token_id] = 0
    mask = data["source_inf_mask"].to(device, dtype=torch.long)
    y[data["target_inf_ids"] == tokenizer.pad_token_id] = -100

    new_inputs={}
    new_inputs["input_ids"]=ids
    new_inputs["attention_mask"]=mask
    new_inputs["labels"]=y

    outputs = model(new_inputs,sent=question_inputs)
    loss_pos = outputs.loss

    optimizer.zero_grad()
    loss_pos.backward()
    optimizer.step()

    return loss_pos.item()


def _score_cands(tokenizer, model, source, cands, question_inputs):
    with torch.no_grad():

        input_ids = tokenizer(source.replace('<mask>', '<extra_id_0>'), return_tensors='pt').input_ids.to(device)
        scores = []
        new_inputs={}
        new_inputs["input_ids"]=input_ids
        for i in range(len(cands)):
            label = tokenizer('<extra_id_0> %s <extra_id_1>' % cands[i], return_tensors='pt').input_ids.to(device)
            new_inputs["labels"]=label
            loss = model(new_inputs,sent=question_inputs).loss.item()
            scores.append(-loss)
        probs = F.softmax(torch.tensor(scores), dim=0)
        return probs


def scores_for_query(tokenizer, model, query, cands, question_inputs):
    scores_, probs_ = [], []
    # source = query
    # scores = _score_cands(tokenizer, model, source, cands)
    # scores_.append(scores)
    scores = _score_cands(tokenizer, model, query, cands, question_inputs)

    return scores



def test(tokenizer_inf, model_inf, test_data):
    num, den = 0, 0
    pbar = tqdm(test_data, total=len(test_data))
    model_inf.eval()
    new_data = []
    
    for item in pbar:
        if 'cands' in item:
            cands = item['cands']
        else:
            cands = ANSWERS
        query = item['query']

        scores = scores_for_query(tokenizer_inf, model_inf, query, cands, query)

        p = scores.argmax().item()
        pred = cands[p]
        item['pred'] = pred

        if 'answer' in item:
            answer = item['answer']
            ok = cmp(answer, pred)
            item['ok'] = ok

        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})

        new_data.append(item)

    return num / den, new_data
        

def cmp(answer, pred):
    if answer == pred:
        return 1
    if answer in ['no', 'zero'] and pred in ['no', 'zero']:
        return 1
    return 0

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


def T5Trainer(ds, args, tokenizer_inf, model_inf, test_data, model_params):

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer_inf = AdamW(params=optimizer_grouped_parameters, lr=model_params["LEARNING_RATE_INF"])
    scheduler_inf = CosineWithRestarts(optimizer_inf, T_max=5, factor=2, eta_min=1e-5)

    # logging
    print("[Data]: Reading data...\n")
    train_data = [ds[i:i + model_params["TRAIN_BATCH_SIZE"]] for i in range(0, len(ds), model_params["TRAIN_BATCH_SIZE"])]
    # Training loop
    print("[Initiating Fine Tuning]...\n")
    with open("result.txt", "w") as f:       
        
        best_acc = 0.0
        for epoch in range(model_params["TRAIN_EPOCHS"]): 
            new_data = []
            loss_all, loss_all_inf = 0.0, 0.0
            for it, iter_data in enumerate(train_data):
                #print(iter_data)
                #raise ValueError
                model_inf.eval()

                iter_data_inf = iter_data
                batch_data_inf = [iter_data_inf[i:i + 1] for i in range(0, len(iter_data_inf), 1)]
                loss_iter_inf = 0
                for batch_ind, batch in enumerate(batch_data_inf):
                    training_set = data_batch_inf(args, model_params, batch, tokenizer_inf)
                    #print(batch)
                    #print(training_set)
                    #raise ValueError
                    loss_inf = train_inf(tokenizer_inf, model_inf, training_set, optimizer_inf, batch[0]["query"])
                    loss_iter_inf += loss_inf
                print("Epoch "+str(epoch)+" Iter "+str(it)+": Inference Loss "+str(loss_iter_inf))
                loss_all += loss_iter_inf
                loss_all_inf += loss_iter_inf

            scheduler_inf.step()

            model_inf.eval()

            acc, new_test_data = test(tokenizer_inf, model_inf, test_data)
            f.write("Epoch " + str(epoch) + " Accuracy: " + str(acc) + "\n")
            print("Epoch " + str(epoch) + " Accuracy: " + str(acc))
            print("Epoch " + str(epoch) + " Loss: " + str(loss_all))
            print("Epoch " + str(epoch) + " INF Loss: " + str(loss_all_inf))
            f.write("Epoch " + str(epoch) + " Loss: " + str(loss_all) + "\n")
            f.write("Epoch " + str(epoch) + " INF Loss: " + str(loss_all_inf) + "\n")

            if acc > best_acc:
                best_acc = acc
                torch.save(model_inf.state_dict(),"model/dscqa_csqa_model.ckpt")
                print("save acc:",best_acc)



def main():
    
    # finetune T5 for inference
    T5Trainer(ds, args, tokenizer_inf, model_inf, test_data, model_params=model_params) 


if __name__ == '__main__':
    main()

