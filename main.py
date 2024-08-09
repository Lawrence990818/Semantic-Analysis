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
    
    def set_seeds(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def fine_tune(self):
        loss_dict = {
            'epoch': [i +1 for i in range(self.epochs)],
            'average training loss':[],
            'avarage validation loss': []
        }

        t0_train = datetime.now()
        for epoch in range(0, self.epochs):
            #train step
            self.model.train()
            training_loss = 0
            t0_epoch  = datetime.now()

            print(f'{"-"*20} Epoch {epoch + 1} {"-"*20}')
            print('\nTarining:\n-------------------')
            print(f'Start time:{t0_epoch}')
            for batch in self.train_dataloader:
                batch_token_ids  = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels  = batch[2].to(self.device)

                self.model.zero_grad()
                loss, logits  = self.model( batch_token_ids, 
                                           token_type_ids = None,
                                           attention_mask=batch_attention_mask, 
                                           labels= batch_labels, return_dict=False)
                training_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimzer.step()
                self.scheduler.step()

            average_train_loss = training_loss / len(self.train_dataloader)
            time_epoch = datetime.now() - t0_epoch

            print(f'Avarage Loss: {average_train_loss}')
            print(f'Time taken: {time_epoch}')

            # validation step
            self.model.eval()
            val_loss = 0
            val_accuracy = 0
            t0_val  = datetime.now()
            print('\nValidation:\n--------------------')
            print(f'Start Time: {t0_val}')

            for batch in self.val_dataloader:
                batch_token_ids = batch[0].to(self.device)
                batch_attention_mask  = batch[1].to(self.device)
                batch_labels =batch[2].to(self.device)

                with torch.no_grad():
                    (loss, logits) = self.model(batch_token_ids,token_type_ids=None,
                                                 attention_mask = batch_attention_mask, 
                                                 labels = batch_labels, return_dict= False )
                    

                logits  = logits.detach().cpu().numpy()
                label_ids  = batch_labels.to('cpu').numpy()
                val_loss += loss.item()
                val_accuracy += self.calculate_accuracy(logits, label_ids)
            average_val_accuracy  = val_accuracy / len(self.val_dataloader)
            average_val_loss  = val_loss/ len(self.val_dataloader)
            time_val  = datetime.now() - t0_val

            print(f'Average Loss: {average_val_loss}')
            print(f'Average Accuracy: {average_val_accuracy}')
            print(f'Time taken: {time_val}\n')

            loss_dict['avarage training loss'].append(average_train_loss)
            loss_dict['avarage validation loss'].append(average_val_loss)

        print(f'Total training time: {datetime.now() - t0_train}') 


    def calculate_accuracy(self, preds, labels):
        pred_flat  = np.argmax(preds, axis= 1).flatten()
        labels_flat  = labels.flatten()
        accuracy = np.sum(pred_flat == labels_flat)/ len(labels_flat)
        return accuracy
    
    def predict(self, dataloader):
        self.model.eval()
        all_logits = []
        
        for batch in dataloader:
            batch_token_ids, batch_attention_mask  = tuple(t.to(self.device) \
                                                            for t in batch)[:2]
            with torch.no_grad():
                logits = self.model(batch_token_ids, batch_attention_mask)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        probs  = F.softmax(all_logits, dim=1).cpu().numpy()
        return probs
        