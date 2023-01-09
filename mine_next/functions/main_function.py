import os.path
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

import csv
import numpy as np
import pandas as pd
import json

# from source.claim_classification.model.modeling import KoElectraForClaimClassification
# from source.claim_classification.func.dataset import convert_data2tensordataset

from mine_next.model.modeling import RobertaForClassification
from mine_next.functions.dataset import (
    convert_data2tensordataset,
    convert_stance_data2tensordataset,
    convert_only_sentence2tensordataset
)


def random_downsampling(dataset):
    major = dataset[dataset['claim_label'] == 'O']
    minor = dataset[dataset['claim_label'] == 'C']
    sampling_data = resample(major, replace=True, n_samples=len(minor)*5, random_state=42)
    train_data = pd.concat([sampling_data, minor])
    return train_data


def random_upsampling(dataset):
    major = dataset[dataset['claim_label'] == 'O']
    minor = dataset[dataset['claim_label'] == 'C']
    sampling_data = resample(minor, replace=True, n_samples=len(major), random_state=42)
    train_data = pd.concat([sampling_data, major])
    return train_data

def do_train(config, model, optimizer, scheduler, train_dataloader, epoch, global_step, total_graph):
    losses = []
    total_predicts, total_corrects = [], []
    for step, batch in tqdm(enumerate(train_dataloader), desc='do_train(epoch_{})'.format(epoch), total=len(train_dataloader)):
        batch = tuple(t.cuda() for t in batch)
        # graph 같이 학습할 경우
        # idx, input_ids, attention_mask, token_type_ids, labels, sim_labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # constituent_labels = batch[6]
        # loss, predicts = model(
        #     idx=idx,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     labels=labels,
        #     sim_labels=sim_labels,
        #     all_graph=total_graph,
        #     constituent_labels=constituent_labels
        # )
        # base
        idx, input_ids, attention_mask, token_type_ids, labels, sim_labels = batch[0], batch[1], batch[2], batch[3], \
                                                                             batch[4], batch[5]
        loss, predicts = model(
            idx=idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            sim_labels=sim_labels,
        )
        predicts = predicts.argmax(dim=-1)
        predicts = predicts.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()

        total_predicts.extend(predicts)
        total_corrects.extend(labels)

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        # 원래는 tensor(0.7255)이런식
        loss.backward()
        losses.append(loss.data.item())
        if (step + 1) % config.gradient_accumulation_steps == 0 or \
                (len(train_dataloader) <= config.gradient_accumulation_steps and (step + 1) == len(
                    train_dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            model.zero_grad()
            global_step += 1
    target_names = ['class 0', 'class 1']
    print(classification_report(total_corrects, total_predicts, target_names=target_names, digits=4))
    accuracy = accuracy_score(total_corrects, total_predicts)
    return accuracy, np.mean(losses), global_step


def do_evaluate(model, dev_dataloader, total_graph):
    total_predicts, total_corrects = [], []
    for step, batch in tqdm(enumerate(dev_dataloader), desc="do_evaluate", total=len(dev_dataloader)):
        batch = tuple(t.cuda() for t in batch)
        # graph 학습할 경우
        # idx, input_ids, attention_mask, token_type_ids, labels, sim_labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # constituent_labels = batch[6]
        # predicts = model(
        #     idx=idx,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     all_graph=total_graph,
        #     constituent_labels=constituent_labels
        # )
        # base
        idx, input_ids, attention_mask, token_type_ids, labels, sim_labels = batch[0], batch[1], batch[2], batch[3], \
                                                                             batch[4], batch[5]
        predicts = model(
            idx=idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        predicts = predicts.argmax(dim=-1)
        predicts = predicts.detach().cpu().tolist()
        labels = labels.detach().cpu().tolist()
        total_predicts.extend(predicts)
        total_corrects.extend(labels)
    target_names = ['class 0', 'class 1']
    print(classification_report(total_corrects, total_predicts, target_names=target_names, digits=4))
    accuracy = accuracy_score(total_corrects, total_predicts)
    return accuracy, total_predicts


def train(config, model, tokenizer):

    # 데이터셋 로드
    train_data = pd.read_csv(config.claim_train, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    train_data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    train_data = train_data.dropna(axis=0)
    # train_data = train_data[:100]
    dev_data = pd.read_csv(config.claim_dev, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    dev_data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    dev_data = dev_data.dropna(axis=0)
    # dev_data = dev_data[:100]

    #train_data = random_upsampling(train_data)
    train_dataset, train_total_graph = convert_only_sentence2tensordataset(train_data, tokenizer, config.max_length, 'train')
    dev_dataset, dev_total_graph = convert_only_sentence2tensordataset(dev_data, tokenizer, config.max_length, 'dev')


    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config.batch_size)

    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epoch
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

    global_step = 0
    max_test_accuracy = 0
    model.zero_grad()
    for epoch in range(config.epoch):
        model.train()
        train_accuracy, average_loss, global_step = do_train(
            config=config, model=model,
            optimizer=optimizer, scheduler=scheduler,
            train_dataloader=train_dataloader, epoch=epoch+1, global_step=global_step, total_graph=train_total_graph)
        print("train_accuracy : {}\taverage_loss : {}\n".format(round(train_accuracy, 4), round(average_loss, 4)))

        model.eval()
        test_accuracy, _ = do_evaluate(model=model, dev_dataloader=dev_dataloader, total_graph=dev_total_graph)
        print("test_accuracy : {}\n".format(round(test_accuracy, 4)))
        output_dir = os.path.join(config.save_dir, "checkpoint-{}".format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(config, os.path.join(output_dir, "training_args.bin"))


def evaluate(config, model, tokenizer):
    dev_data = pd.read_csv(config.claim_dev, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    dev_data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    dev_data = dev_data.dropna(axis=0)
    # dev_data = dev_data[:10]
    # dev_dataset = convert_stance_data2tensordataset(dev_data, tokenizer, config.max_length)
    dev_dataset, total_graph = convert_only_sentence2tensordataset(dev_data, tokenizer, config.max_length, 'dev')

    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config.batch_size)

    test_accuracy, total_predicts = do_evaluate(model=model, dev_dataloader=dev_dataloader, total_graph=total_graph)
    print("test accuracy : {}".format(round(test_accuracy,4)))
    total_corrects = dev_data['claim_label'].tolist()
    total_corrects = [1 if correct == 'C' else 0 for correct in total_corrects]
    totaL_claim_sentence = dev_data['claim_sentence'].tolist()
    error_list = []
    for predict, correct, claim in zip(total_predicts, total_corrects, totaL_claim_sentence):
        if predict != correct:
            error = {}
            error['predict'] = predict
            error['correct'] = correct
            error['claim_sentence'] = claim
            error_list.append(error)

    with open('../mine/functions/dev_error.json', 'w', encoding='utf-8') as f:
        json.dump(error_list, f, indent=4)
