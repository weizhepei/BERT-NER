# BERT-NER

**This project implements an solution to the "X" label issue (e.g., [#148](https://github.com/huggingface/pytorch-transformers/issues/148), [#422](https://github.com/huggingface/pytorch-transformers/issues/422)) of NER task in Google's BERT [paper](https://arxiv.org/pdf/1810.04805.pdf), and is developed mostly based on lemonhu's [work](https://github.com/lemonhu/NER-BERT-pytorch) and bheinzerling's [suggestion](https://github.com/huggingface/pytorch-transformers/issues/64#issuecomment-443703063).**

## Dataset

- Chinese: [MSRA](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra), which is [reported](https://github.com/lemonhu/NER-BERT-pytorch/issues/9) to be incomplete. A complete version can be found [here](https://github.com/buppt/ChineseNER/tree/master/data/MSRA).
- English: [CONLL-2003](https://github.com/kyzhouhzau/BERT-NER/tree/master/data)

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.0.1. The requirements are:

- tensorflow >= 1.11.0
- pytorch >= 1.0.1
- pytorch-transformers == 1.0.0
- tqdm
- apex

**Note**: The tensorflow library is only used for the conversion of pre-trained models from TensorFlow to PyTorch. apex is a tool for easy mixed precision and distributed training in Pytorch, please see https://github.com/NVIDIA/apex.

## Usage

1. **Get BERT model for PyTorch**

   - **Convert the TensorFlow checkpoint to a PyTorch dump**

     - Download the Google's BERT pretrained models for Chinese  **[(`BERT-Base, Chinese`)](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)** and English **[(`BERT-Base, Cased`)](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**. Then decompress them under `pretrained_bert_models/bert-chinese-cased/` and `pretrained_bert_models/bert-base-cased/` respectively. More pre-trained models are available [here](https://github.com/google-research/bert#pre-trained-models).

     - Execute the following command,  convert the TensorFlow checkpoint to a PyTorch dump as huggingface [suggests](https://huggingface.co/pytorch-transformers/converting_tensorflow_models.html). Here is an example of the conversion process for a pretrained `BERT-Base Cased` model.
       ```shell
       export TF_BERT_MODEL_DIR=/full/path/to/cased_L-12_H-768_A-12
       export PT_BERT_MODEL_DIR=/full/path/to/pretrained_bert_models/bert-base-cased
    
       pytorch_transformers bert \
       	$TF_BERT_MODEL_DIR/bert_model.ckpt \
    	   $TF_BERT_MODEL_DIR/bert_config.json \
       	$PT_BERT_MODEL_DIR/pytorch_model.bin
       ```
   
     - Copy the BERT parameters file `bert_config.json` and dictionary file `vocab.txt` to the directory `$PT_BERT_MODEL_DIR`.
       ```
       cp $TF_BERT_MODEL_DIR/bert_config.json $PT_BERT_MODEL_DIR/config.json
       cp $TF_BERT_MODEL_DIR/vocab.txt $PT_BERT_MODEL_DIR/vocab.txt
       ```
   
2. **Build dataset and tags**

   if you use default parameters (using CONLL-2003 dataset as default) , just run

   ```shell
   python build_dataset_tags.py
   ```

   Or specify dataset (e.g., MSRA) and other parameters on the command line

   ```shell
   python build_dataset_tags.py --dataset=msra
   ```

   It will extract the sentences and tags from `train_bio`, `test_bio` and `val_bio`(if provided, otherwise randomly sample 5% data from the `train_bio`). Then split them into train/val/test and save them in a convenient format for our model, and create a file `tags.txt` containing a collection of tags.

3. **Set experimental hyperparameters**

   We created directories with the same name as datasets under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like

   ```json
   {
       "full_finetuning": true,
       "max_len": 180,
   
       "learning_rate": 5e-5,
       "weight_decay": 0.01,
       "clip_grad": 5,
   }
   ```

   For different datasets, you will need to create a new directory under `experiments` with  `params.json`.

4. **Train and evaluate the model**

   if you use default parameters (using CONLL-2003 dataset as default) , just run

   ```python
   python train.py
   ```

   Or specify dataset (e.g., MSRA) and other parameters on the command line

   ```shell
   python train.py --dataset=msra --multi_gpu
   ```

   A proper pretrained BERT model will be automatically chosen according to the language of the specified dataset. It will instantiate a model and train it on the training set following the hyper-parameters specified in `params.json`. It will also evaluate some metrics on the development set.

5. **Evaluation on the test set**

   Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set.

   if you use default parameters (using CONLL-2003 dataset as default) , just run

   ```shell
   python evaluate.py
   ```

   Or specify dataset (e.g., MSRA) and other parameters on the command line

   ```shell
   python evaluate.py --dataset=msra
   ```
