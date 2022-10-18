import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score




class MLMDataset(Dataset):

  def __init__(self, data, tokenizer, max_length, mask_ratio):
    self.data = data
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.mask_ratio = mask_ratio

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    text = self.data[idx]
    inputs = self.tokenizer(
          text,
          max_length=self.max_length,
          padding='max_length',
          truncation=True,
          return_tensors='pt')
    
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < self.mask_ratio) * (inputs.input_ids != self.tokenizer.cls_token_id) * \
                                          (inputs.input_ids != self.tokenizer.sep_token_id) * \
                                          (inputs.input_ids != self.tokenizer.pad_token_id)
    tokens_2_mask = mask_arr.view(-1).nonzero().view(-1)
    labels = inputs.input_ids[0, tokens_2_mask]
    inputs.input_ids[0, tokens_2_mask] = self.tokenizer.mask_token_id
    inputs['labels'] = labels.view(-1)
    return inputs
  
  @staticmethod
  def collate(inputs):
    dict_ = dict()
    dict_['labels'] = torch.cat([input.pop('labels') for input in inputs])
    for key in inputs[0].keys():
      dict_[key] = torch.stack([input[key] for input in inputs], dim=0).squeeze()
    dict_['masked_token_indexes'] = (dict_['input_ids'].view(-1) == 103).nonzero().view(-1)
    return dict_

def tolist(tensor):
  return tensor.view(-1).detach().cpu().tolist()

def score(labels, preds):
  f1_score_ = f1_score(labels, preds, average='micro')
  recall_score_ = recall_score(labels, preds, average='micro')
  precision_score_ = precision_score(labels, preds, average='micro')
  print('--------------- Scores ---------------')
  print(f'f1 score -> {round(f1_score_, 3)}')
  print(f'recall score -> {round(recall_score_, 3)}')
  print(f'precision score -> {round(precision_score_, 3)}')