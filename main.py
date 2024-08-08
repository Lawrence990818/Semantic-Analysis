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
    
    def tokenize_dataset(self):
        token_ids = []
        attention_masks = []

        for review in self.df_dataset['review_cleaned']:
            tokens, masks  = self.tokenize(review)
            token_ids.append(tokens)
            attention_masks.append(masks)

        token_ids  = torch.cat(token_ids, dim= 0)
        attention_masks  = torch.cat(attention_masks, dim=0)
        return token_ids, attention_masks
    
    def create_dataloader(self):
        train_ids, val_ids  = train_test_split(self.token_ids, test_size = self.val_size, shuffle=False)
        train_masks, val_masks  = train_test_split(self.attention_masks, test_size = self.val_size, shuffle=False)

        labels =torch.tensor(self.df_dataset['sentiment_encoded'].values)
        train_labels, val_labels =train_test_split(labels, test_size = self.val_size, shuffle=False)
        train_data  = TensorDataset(train_ids, train_masks, train_labels)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
        val_data  = TensorDataset(val_ids, val_masks, val_labels)
        val_dataloader  = DataLoader(val_data, batch_size=16)

        return train_dataloader, val_dataloader

    def creates_scheduler(self):
        num_training_steps = self.epochs* len(self.train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            self.optimzer,
            num_warmup_steps=0,
            num_training_step = num_training_steps
        )
        return scheduler