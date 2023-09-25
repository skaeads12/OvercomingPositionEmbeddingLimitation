
import json
from argparse import ArgumentParser

from model import BertForQuestionAnswering
from dataset import QADataset

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-td", "--test_dir", type=str, default="data/korquad1.0/test.json")
    parser.add_argument("-cd", "--context_dir", type=str, default="data/korquad1.0/context.json")
    parser.add_argument("-sd", "--save_dir", type=str, default="result/korquad1.0/")

    parser.add_argument("-pc", "--pretrained_config", type=str, default="klue/bert-base")
    parser.add_argument("-pt", "--pretrained_tokenizer", type=str, default="result/korquad1.0/checkpoint-850")
    parser.add_argument("-pm", "--pretrained_model", type=str, default="result/korquad1.0/checkpoint-850")

    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-ml", "--max_length", type=int, default=512)

    return parser.parse_args()

def compute_metrics(preds):

    predictions = preds.predictions
    label_ids = preds.label_ids

    

    pass

def predict(args):

    context = json.load(open(args.context_dir, 'r'))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)

    test_set = QADataset(
        data_dir = args.test_dir,
        context = context,
        tokenizer = tokenizer,
        max_length = args.max_length,
    )

    model = BertForQuestionAnswering.from_pretrained(args.pretrained_model)

    train_args = TrainingArguments(
        output_dir = args.save_dir,
        overwrite_output_dir = True,
        do_predict = True,
        per_device_eval_batch_size = args.batch_size,
        logging_steps = 10,
        seed = 42,
        data_seed = 42,
    )

    trainer = Trainer(
        model = model,
        args = train_args,
        tokenizer = tokenizer,
    )

    eval_preds = trainer.predict(test_dataset=test_set)

    predictions = eval_preds.predictions
    label_ids = eval_preds.label_ids

    print(predictions.shape)
    print(label_ids[0].shape)
    print(label_ids[1].shape)

if __name__=="__main__":

    args = parse_args()
    predict(args)
