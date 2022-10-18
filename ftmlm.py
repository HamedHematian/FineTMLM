from utils import *
import random
import torch
from glob import glob
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForMaskedLM
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, RMSprop
from copy import deepcopy
from tqdm import tqdm


class FineTMLM:

  def __init__(self,
               data,
               mlm_model,
               tokenizer,
               batch_size=20,
               max_length=100,
               learning_rate=1e-5,
               warmup_ratio=0.,
               epochs=10,
               do_eval=True,
               eval_ratio=.1,
               do_log=True,
               log_step=100,
               device='cuda:0',
               checkpoint_dir='.',
               mask_ratio=.15):
    
    self.divide_train_eval(data, do_eval, eval_ratio)
    self.mlm_model = mlm_model.to(device)
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.max_length = max_length
    self.learning_rate = learning_rate
    self.warmup_ratio = warmup_ratio
    self.epochs = epochs
    self.do_eval = do_eval
    self.do_log = do_log
    self.log_step = log_step
    self.device = device
    self.checkpoint_dir = checkpoint_dir
    self.mask_ratio = mask_ratio
    self.initialize_optimizer()


  def divide_train_eval(self, data, do_eval, eval_ratio):
    random.shuffle(data)
    if do_eval:
      train_ratio = 1 - eval_ratio
      data_len = len(data)
      train_data_cut = int(train_ratio * data_len)
      self.train_data, self.eval_data = data[: train_data_cut], data[train_data_cut: ]
    else:
      self.train_data = data


  def initialize_optimizer(self):
    self.train_dataset = MLMDataset(self.train_data, self.tokenizer, self.max_length, self.mask_ratio)
    self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=MLMDataset.collate)
    if self.do_eval:
      self.eval_dataset = MLMDataset(self.eval_data, self.tokenizer, self.max_length, self.mask_ratio)
      self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=MLMDataset.collate)
    self.loss_fn = nn.CrossEntropyLoss()
    self.loss_collection = list()
    self.optimizer = AdamW(self.mlm_model.parameters(), lr=self.learning_rate)
    optimization_steps = self.epochs * len(self.train_dataloader)
    warmup_steps = int(optimization_steps * self.warmup_ratio)
    self.scheduler = get_linear_schedule_with_warmup(
      optimizer=self.optimizer,
      num_warmup_steps=warmup_steps, 
      num_training_steps=optimization_steps)
      

  def save_checkpoint(self, epoch):
    filename = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}')
    print(filename)
    checkpoint_config = {
      'epoch': epoch,
      'optimizer_dict': self.optimizer.state_dict(),
      'scheduler_dict': self.scheduler.state_dict(),
      'model_dict': self.mlm_model.state_dict()}
    torch.save(checkpoint_config, filename)

  
  def load_checkpoint(self):
    checkpoint_files = glob('checkpoint*')
    self.checkpoint_available = True if checkpoint_files != [] else False
    if self.checkpoint_available:
      current_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1]))[-1]
      self.checkpoint_config = torch.load(current_checkpoint)
      self.mlm_model.load_state_dict(self.checkpoint_config['model_dict']),
      self.optimizer.load_state_dict(self.checkpoint_config['optimizer_dict'])
      self.scheduler.load_state_dict(self.checkpoint_config['scheduler_dict'])



  def log(self, epoch, step):
    if len(self.loss_collection) % self.log_step == 0:
      print(f'EPOCH [{epoch + 1}/{self.epochs}] | STEP [{step + 1}/{len(self.train_dataloader)}] | Loss {round(sum(self.loss_collection) / len(self.loss_collection), 2)}')
      self.loss_collection = list()
      

  def train(self, epoch):
    self.mlm_model.train()
    print(f'---------- Training on {self.device} ----------')
    for step, data in enumerate(self.train_dataloader):
      for key in data.keys():
        data[key] = data[key].to(self.device)
      # run network
      masked_token_indexes = data.pop('masked_token_indexes')
      labels = data.pop('labels')
      logits = self.mlm_model(**data).logits.view(-1, len(self.tokenizer))
      masked_logits = logits[masked_token_indexes, :]
      # loss
      loss = self.loss_fn(masked_logits, labels)
      # optimize
      loss.backward()
      self.loss_collection.append(loss.item())
      self.optimizer.step()
      self.scheduler.step()
      self.loss_collection.append(loss.item())
      self.log(epoch, step)


  def eval(self, epoch):
    self.mlm_model.eval()
    all_preds = list()
    all_labels = list()
    for step, data in tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader)):
      for key in data.keys():
        data[key] = data[key].to(self.device)
      # run network
      masked_token_indexes = data.pop('masked_token_indexes')
      labels = tolist(data.pop('labels'))
      logits = self.mlm_model(**data).logits.view(-1, len(self.tokenizer))
      masked_logits = logits[masked_token_indexes, :]
      maked_pred = tolist(masked_logits.argmax(1))
      all_labels.extend(labels)
      all_preds.extend(maked_pred)
    score(all_labels, all_preds)

  def __call__(self):
    self.load_checkpoint()
    start_epoch = self.checkpoint_config['epoch'] if self.checkpoint_available else 0
    for epoch in range(start_epoch, self.epochs):
      self.train(epoch)
      if self.do_eval:
        self.eval(epoch)
      self.save_checkpoint(epoch + 1)
  

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

text = [
    'here is some text to train on',
    'it must be a list of sentences',
    'easy to pick some data in your domain'
] * 100

ftm = FineTMLM(text, model, tokenizer)
ftm()