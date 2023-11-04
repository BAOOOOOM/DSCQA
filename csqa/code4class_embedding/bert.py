# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM


class Config(object):

    """配置参数"""
    def __init__(self, args):
        self.model_name = 'bert'
        self.rank = -1
        self.local_rank = -1
        self.train_path = args.data_dir + '/csqa_train.json'  # 训练集
        self.test_path = args.data_dir + '/awe'  # 测试集
        self.save_path = args.output_dir  # 模型训练结果
        self.bert_path = args.model_dir
        self.test_batch = args.test_batch
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_workers = 1
        self.local_rank = -1
        self.num_classes = 2                         # 类别数
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                           # mini-batch大小
        self.learning_rate = args.learning_rate                                     # 学习率
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.hidden_size = args.hidden_size
        self.max_length = args.max_length
