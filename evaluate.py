"""Evaluate the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

from SequenceTagger import BertForSequenceTagging, BertConfig

from metrics import f1_score
from metrics import classification_report

from data_loader import DataLoader
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")


def evaluate(model, data_iterator, params, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data, batch_token_starts, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)
        
        loss = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())
        
        batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]  # shape: (batch_size, max_len, num_labels)
        
        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()

        pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])
        true_tags.extend([[idx2tag.get(idx) if idx != -1 else 'O' for idx in indices] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


if __name__ == '__main__':
    args = parser.parse_args()
    
    tagger_model_dir = 'experiments/' + args.dataset
    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset
    if args.dataset in ["conll"]:
        bert_model_dir = 'pretrained_bert_models/bert-base-cased/'
    elif args.dataset in ["msra"]:
        bert_model_dir = 'pretrained_bert_models/bert-base-chinese/'

    data_loader = DataLoader(data_dir, bert_model_dir, params, token_pad_idx=0, tag_pad_idx=-1)


    # Load data
    test_data = data_loader.load_data('test')

    # Specify the test set size
    params.test_size = test_data['size']
    params.eval_steps = params.test_size // params.batch_size
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

    logging.info("- done.")

    # Define the model
    # config_path = os.path.join(args.bert_model_dir, 'config.json')
    # config = BertConfig.from_json_file(config_path)
    ## model = BertForTokenClassification(config, num_labels=len(params.tag2idx))
    # model = BertForSequenceTagging(config)
    model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    model.to(params.device)
    
    if args.fp16:
        model.half()
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("Starting evaluation...")
    test_metrics = evaluate(model, test_data_iterator, params, mark='Test', verbose=True)

