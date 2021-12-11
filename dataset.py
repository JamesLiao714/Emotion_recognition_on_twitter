import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class TweetDataset(Dataset):
    def __init__(self, df, mode, tokenizer):
        assert mode in ["train", "test"]  
        self.mode = mode
        if mode == "train":
            self.df = df[['text', 'emotion']].reset_index(drop = True)
        elif mode == "test":
            self.df = df[['tweet_id', 'text']].reset_index(drop = True)
        self.len = len(self.df)
        self.label_map = {'anticipation': 0, 'joy': 1, 'disgust': 2, 'sadness': 3, 'trust': 4, 'fear': 5, 'surprise': 6, 'anger': 7}
        self.tokenizer = tokenizer 
    
    def __getitem__(self, idx):
        tweet = self.df.iloc[idx]
        encoded = self.tokenizer.encode_plus(tweet.text,
                                            add_special_tokens = True,
                                            return_attention_mask = True,
                                            pad_to_max_length = True,
                                            max_length = 256,
                                            return_tensors = 'pt')
        if self.mode == 'train':
            labels = self.label_map[tweet.emotion]
        input_ids = encoded['input_ids'][0]
        attention_masks = encoded['attention_mask'][0]
        if self.mode == "test":
            return (input_ids, attention_masks)
        elif self.mode == "train":
            return (input_ids, attention_masks, labels)
    
    def __len__(self):
        return self.len
