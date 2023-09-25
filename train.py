

import re
import json
import string
from collections import Counter
from argparse import ArgumentParser

from model import BertForQuestionAnswering, RobertaForQuestionAnswering
from dataset import QADataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-td", "--train_dir", type=str, default="data/korquad1.0/train.json")
    parser.add_argument("-vd", "--eval_dir", type=str, default="data/korquad1.0/eval.json")
    parser.add_argument("-cd", "--context_dir", type=str, default="data/korquad1.0/context.json")
    parser.add_argument("-sd", "--save_dir", type=str, default="result/korquad1.0_1024/")

    parser.add_argument("-pt", "--pretrained_tokenizer", type=str, default="klue/bert-base")
    parser.add_argument("-pm", "--pretrained_model", type=str, default="klue/bert-base")

    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("-p", "--patience", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-ml", "--max_length", type=int, default=1024)

    return parser.parse_args()

def compute_metrics(preds):

    predictions = preds.predictions
    label_ids = preds.label_ids

    pred_start = predictions[0].argmax(-1)
    pred_end = predictions[1].argmax(-1)

    label_start = label_ids[0]
    label_end = label_ids[1]

    total = 0
    exact_match = 0

    for p_start, p_end, l_start, l_end in zip(pred_start, pred_end, label_start, label_end):
        
        if p_start == l_start and p_end == l_end:
            exact_match += 1
        total += 1

    return {"exact_match": exact_match / total}

def train(args):

    context = json.load(open(args.context_dir, 'r'))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)

    train_set = QADataset(
        data_dir = args.train_dir,
        context = context,
        tokenizer = tokenizer,
        max_length = args.max_length,
    )

    eval_set = QADataset(
        data_dir = args.eval_dir,
        context = context,
        tokenizer = tokenizer,
        max_length = args.max_length,
    )

    config = AutoConfig.from_pretrained(args.pretrained_model)
    if config.model_type == "bert":
        model = BertForQuestionAnswering.from_pretrained(args.pretrained_model)
        model.set_position_embeddings(args.max_length)
    elif config.model_type == "roberta":
        model = RobertaForQuestionAnswering.from_pretrained(args.pretrained_model)
        model.set_position_embeddings(args.max_length + 2)

    train_args = TrainingArguments(
        output_dir = args.save_dir,
        overwrite_output_dir = True,
        do_train = True,
        do_eval = True,
        evaluation_strategy = "epoch",
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.num_epochs,
        logging_steps = 10,
        save_strategy = "epoch",
        save_total_limit = 3,
        seed = 42,
        data_seed = 42,
        load_best_model_at_end = True,
        metric_for_best_model = "exact_match",
        greater_is_better = True,
    )

    trainer = Trainer(
        model = model,
        args = train_args,
        train_dataset = train_set,
        eval_dataset = eval_set,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    trainer.train()

if __name__=="__main__":

    args = parse_args()
    train(args)
