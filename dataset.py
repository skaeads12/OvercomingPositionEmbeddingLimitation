
import re
import json

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

class QADataset(Dataset):

    def __init__(
        self,
        data_dir: str = None,
        context: list = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
    ):
        self.samples = json.load(open(data_dir, 'r'))
        self.context = context
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample["question"]
        answer_start = sample["answer_start"]
        answer_end = sample["answer_end"]
        context = self.context[sample["context"]]

        answer_tokens = self.tokenizer.tokenize(context[answer_start:answer_end])
        context_tokens = self.tokenizer.tokenize(context)
        labels = []

        for i in range(len(context_tokens) - len(answer_tokens)):
            if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
                labels = [i, i+len(answer_tokens)]
                break

        question_ids = self.tokenizer(question).input_ids
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)

        input_ids = question_ids + context_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(question_ids) + [1] * (len(context_ids) + 1)

        if len(input_ids) < self.max_length:
            gap = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * gap
            attention_mask += [0] * gap
            token_type_ids += [0] * gap

        labels = [label + len(question_ids) if label + len(question_ids) < self.max_length else -100 for label in labels]

        if len(labels) == 0 or self.tokenizer.unk_token in answer_tokens:
            labels = [-100, -100]

        return {
            "input_ids": torch.LongTensor(input_ids[:self.max_length]),
            "attention_mask": torch.LongTensor(attention_mask[:self.max_length]),
            "token_type_ids": torch.LongTensor(token_type_ids[:self.max_length]),
            "start_positions": labels[0],
            "end_positions": labels[1],
        }
    
if __name__=="__main__":

    from tqdm import tqdm
    from time import time
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    data_dir = "data/aihub_mrc/train.json"
    context_dir = "data/aihub_mrc/context.json"
    max_length = 512
    
    context = json.load(open(context_dir, 'r'))
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    dataset = QADataset(data_dir=data_dir, context=context, tokenizer=tokenizer, max_length=max_length)
    
    err_cnt = 0
    start = time()

    for sample in tqdm(dataset):
        if sample["start_positions"] == -100 or sample["end_positions"] == -100:
            err_cnt += 1

    print("dataset load in {} (s).".format(time() - start))
    print("-100 ratio is {}/{} ({} %)".format(err_cnt, len(dataset), err_cnt / len(dataset) * 100))
