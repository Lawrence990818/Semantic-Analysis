from datetime import datetime
import pandas as np
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import adamw
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

class FineTuningPipeline:
    def __init__(self, dataset, tokenizer,
                  model, optimizer, loss_function= nn.CrossEntropyLoss(),
                    val_size=0.1, epochs =4, seed=42):
        self.df_dataset = dataset
        self.tokenizer = tokenizer
        self.model  = model
        self.optimzer = optimizer
        self.loss_function = loss_function
        self.val_size  = val_size
        self.epochs  = epochs
        self.seed  = seed


        ## check if GPU is available  for faster training
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device  = torch.device('cpu')
        

        #perfrom fine-tuning
        self.model.to(self.device)
        self.set_seeds()
        self.token_ids, self.attention_masks = self.tokenize_dataset()
        self.train_dataloader, self.val_dataloader = self.create_dataloaders()
        self.scheduler = self.create_scheduler()
        self.fine_tune()


    def tokenize(self, text):
        batch_encoder  = self.tokenizer.encode_plus(
            text,
            max_length = 512,
            padding= 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )  
        token_ids = batch_encoder['input_ids']
        attention_mask  = batch_encoder['attention_mask']
        return token_ids, attention_mask
    
    