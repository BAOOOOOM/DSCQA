# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation,
# Which we do not include in it.

import argparse
import torch
from  torch import nn
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F

#device=get_device()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 
Batch_Size=32
predict_id=0


parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-large')
parser.add_argument("--train_dataset_path", default='8class_data/csqa2_train.json')
parser.add_argument("--dev_dataset_path", default='8class_data/csqa2_dev.json')
parser.add_argument("--test_dataset_path", default='8class_data/csqa2_test.json')
parser.add_argument("--save_path", default='prefix_test_output/')
args = parser.parse_args()
print(args)

from code4class_embedding.sentence_t5_get_embedding import get_class_embedding_entry,get_question_embedding_list,get_question_embedding

class_embedding_dict=get_class_embedding_entry(device)
print("get class embedding successful!")
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
guid2question_dict={}

#改写
from openprompt.data_utils import InputExample
import json
train_dataset=[]
with open(args.train_dataset_path) as f:
    lines=f.readlines()
    for line in lines:
        data=json.loads(line)
        question=data["question"]
        answer=data["answer"]
        id=data["id"]
        question_class=data["question_class"]
        label=0
        if answer=='yes':
            label=1
        input_example = InputExample(text_a = question, label=label, guid=id)
        train_dataset.append(input_example)
        guid2question_dict[id]=SEP.join([question])

dev_dataset=[]
with open(args.dev_dataset_path) as f:
    lines=f.readlines()
    for line in lines:
        data=json.loads(line)
        question=data["question"]
        answer=data["answer"]
        id=data["id"]
        label=0
        if answer=='yes':
            label=1
        input_example = InputExample(text_a = question, label=label, guid=id)
        dev_dataset.append(input_example)
        guid2question_dict[id]=SEP.join([question])

test_dataset=[]
with open(args.test_dataset_path) as f:
    lines=f.readlines()
    for line in lines:
        data=json.loads(line)
        question=data["question"]
        #answer=data["answer"]
        id=data["id"]
        label=0
        input_example = InputExample(text_a = question, label=label, guid=id)
        test_dataset.append(input_example)
        guid2question_dict[id]=SEP.join([question])

print("load dataset successful!")

# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"} ', using_decoder_past_key_values=True,num_token=100,prefix_dropout=0.5)


# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=train_dataset, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=3,
    batch_size=Batch_Size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dev_dataset, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=3,
    batch_size=Batch_Size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=test_dataset, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=3,
    batch_size=Batch_Size,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

from openprompt.prompts import ManualVerbalizer

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[["no"], ["yes"]])



# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
from openprompt import PromptForClassification



from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.
loss_func = torch.nn.CrossEntropyLoss()

#print(mytemplate.named_parameters())

from transformers.optimization import get_linear_schedule_with_warmup

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function

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

        return self.output_linear(x)


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=True, plm_eval_mode=args.plm_eval_mode)
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
        a_emb,question_embedding,question_seq_emb=get_question_embedding(sent,device,Batch_Size)
        if question_embedding.size()[0]!=Batch_Size:
            cat_class_embedding=get_cat_class_emb(class_embedding_dict,question_embedding.size()[0])
        else:
            cat_class_embedding=normal_cat_class_embedding
        
        now_batch=question_embedding.size()[0]

        question_embedding=question_embedding.expand(1,question_embedding.size()[0],question_embedding.size()[1])

        ###################get dynamic prefixes######################
        dynamic_class=self.multihead_attn(question_embedding,cat_class_embedding,cat_class_embedding)
        #print(type(dynamic_class))
        dynamic_class=dynamic_class.permute(1,0,2)

        trans_cat_class_embedding=self.skill_emb_trans(cat_class_embedding)
        trans_cat_class_embedding=trans_cat_class_embedding.permute(1,0,2)

        logits = self.prompt_model(inputs,mulit_prefix=dynamic_class,cat_class_embedding=trans_cat_class_embedding)
        return logits
    

model=Mymodel().to(device)

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
    "params": [p for n, p in model.multihead_attn.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model.multihead_attn.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model.skill_emb_trans.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model.skill_emb_trans.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model.prompt_model.plm.skill2question_multihead_attn.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in model.prompt_model.plm.skill2question_multihead_attn.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

def evaluate(model, dataloader):
    model.eval()

    allpreds = []
    alllabels = []
    allquestion=[]
    allid=[]
    tot_loss=0
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            question_inputs=[]
            for id in inputs["guid"]:
                question_inputs.append(guid2question_dict[id])
            logits = model(inputs,question_inputs)
            labels = inputs['label']
            allid.extend(inputs["guid"])
            allquestion.extend(question_inputs)
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            loss = loss_func(logits, labels)
            tot_loss+=loss.item()

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("dev_loss:",loss.item())

    return acc,allid,allquestion,alllabels,allpreds

def train():
    print("start to train!")
    # training and generation.
    global_step = 0
    tot_loss = 0
    log_loss = 0
    best_acc=0

    for epoch in range(1500):
        model.train()
        print("epoch:",epoch)
        for step, inputs in enumerate(train_dataloader):
            global_step +=1
            inputs = inputs.to(device)
            question_inputs=[]
            for id in inputs["guid"]:
                question_inputs.append(guid2question_dict[id])
            logits = model(inputs,question_inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()
        print("Epoch {}, average loss: {}".format(epoch, loss.item()), flush=True)
        nowacc,all_devid,all_devquestion,all_devlabels,all_devpreds = evaluate(model, validation_dataloader)
        print("the accuracy:",nowacc)
        if nowacc>best_acc:
            best_acc=nowacc
            torch.save(model.state_dict(),"model/dscqa_csqa2_model.ckpt")
            print("save accuracy:",best_acc)



if __name__ == '__main__':
    train()
