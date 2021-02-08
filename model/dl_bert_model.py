import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
model = BertForSequenceClassification.from_pretrained("bert-base-uncased") 

tokenizer.save_pretrained('./bert_base') 
model.save_pretrained('./bert_base')