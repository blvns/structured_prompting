from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F
from sklearn import metrics

import os
import random
import numpy as np

#code adapted from https://github.com/mallorbc/GPTNeoX20B_HuggingFace/blob/main/main.py
#Note: this only works with two devices (48Gb) available and ~72Gb memory
def init_gpt_neox(fp16=False):
	model_name = "EleutherAI/gpt-neox-20b"
	weights_path = "/gscratch/scrubbed/blvns/gptneox"
	if not os.path.exists(weights_path):
	    os.makedirs(weights_path)
	    if fp16:
	    	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
	    else:
	        model = AutoModelForCausalLM.from_pretrained(model_name)

	    model.save_pretrained(weights_path)

	config = AutoConfig.from_pretrained(model_name)

	config.use_cache = False

	with init_empty_weights():
	    model = AutoModelForCausalLM.from_config(config)

	#this was returning an empty device map, so hardcoding one for GPT-NeoX
	#device_map = infer_auto_device_map(model, no_split_module_classes=["GPTNeoXLayer"], dtype=torch.float16)
	device_map = {
		'gpt_neox.embed_in': 0,
		'gpt_neox.layers.0': 0,
		'gpt_neox.layers.1': 0,
		'gpt_neox.layers.2': 0,
		'gpt_neox.layers.3': 0,
		'gpt_neox.layers.4': 0,
		'gpt_neox.layers.5': 0,
		'gpt_neox.layers.6': 0,
		'gpt_neox.layers.7': 0,
		'gpt_neox.layers.8': 0,
		'gpt_neox.layers.9': 0,
		'gpt_neox.layers.10': 0,
		'gpt_neox.layers.11': 0,
		'gpt_neox.layers.12': 0,
		'gpt_neox.layers.13': 0,
		'gpt_neox.layers.14': 0,
		'gpt_neox.layers.15': 0,
		'gpt_neox.layers.16': 0,
		'gpt_neox.layers.17': 0,
		'gpt_neox.layers.18': 0,
		'gpt_neox.layers.19': 0,
		'gpt_neox.layers.20': 0,
		'gpt_neox.layers.21': 0,
		'gpt_neox.layers.22': 1,
		'gpt_neox.layers.23': 1,
		'gpt_neox.layers.24': 1,
		'gpt_neox.layers.25': 1,
		'gpt_neox.layers.26': 1,
		'gpt_neox.layers.27': 1,
		'gpt_neox.layers.28': 1,
		'gpt_neox.layers.29': 1,
		'gpt_neox.layers.30': 1,
		'gpt_neox.layers.31': 1,
		'gpt_neox.layers.32': 1,
		'gpt_neox.layers.33': 1,
		'gpt_neox.layers.34': 1,
		'gpt_neox.layers.35': 1,
		'gpt_neox.layers.36': 1,
		'gpt_neox.layers.37': 1,
		'gpt_neox.layers.38': 1,
		'gpt_neox.layers.39': 1,
		'gpt_neox.layers.40': 1,
		'gpt_neox.layers.41': 1,
		'gpt_neox.layers.42': 1,
		'gpt_neox.layers.43': 1,
		'gpt_neox.final_layer_norm':1,
		'embed_out': 1
	}
	model = load_checkpoint_and_dispatch(
	    model,
	    weights_path,
	    device_map=device_map,
	    offload_folder=None,
	    offload_state_dict=False,
	    dtype="float16"
	)

	return model

def _load_ud(file_path, task):
	dataset = []
	example_sent = []
	example_labels = []
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#'): continue
			if len(line) == 0:
				if task == 'udp':
					p = []
					for l1, l2 in example_labels:
						if l1 == -1: p.append(("NONE", "none", l1))
						elif l1 == 0: p.append(("ROOT", "root", l1))
						else: p.append((example_sent[l1-1], l2, l1))
					example_labels = p
					x, y, z = zip(*example_labels)
					example_labels = [list(x), list(y), list(z)]
				dataset.append((example_sent, example_labels))
				example_sent = []
				example_labels = []
			else:
				idx, word, lemma, upos, xpos, morph_feats, head, dep_rel, deps, misc = line.split('\t')
				#skip mulitpart phrases since each part is additionally annotated 
				#(weird for tokenization but ¯\_(ツ)_/¯)
				if '-' in idx: continue
				if idx == 1: assert len(example) == 0
				#using upos for part of speech task instead of xpos
				if head != '_': head = int(head)
				else: head = -1
				if task == 'pos': label = upos
				#stripping the detailed label
				elif task == 'dep_rel': label = dep_rel.split(':')[0]
				else: label = (head, dep_rel)
				example_sent.append(word)
				example_labels.append(label)
	if len(example_sent) > 0: dataset.append((example_sent, example_labels))
	return dataset

def load_ud_datasets(data_path, lang, task='pos', split='train'):
    ud_files = os.listdir(data_path)
    data_file = [u for u in ud_files if u.startswith(lang+'_') and '-{}'.format(split) in u]
    #print(data_file)

    assert len(data_file) == 1 
    data_file = data_file[0]

    data_path = os.path.join(data_path, data_file)
    ud_data = _load_ud(data_path, task)

    return ud_data

def process_ontonotes(data, label_space, task):
	if label_space == None:
		label_space = set()

	cleaned_data = []
	for doc in data:
		for sent in doc['sentences']:
			tokens = sent['words']
			if task == 'ner':
				labels = [label_space[i] for i in sent['named_entities']]
				cleaned_data.append((tokens, labels))
			#elif task == 'srl':
				#different example for each predicate in the sentence
			#	for frame in sent['srl_frames']:
			#		predicate = frame['verb']
			#		labels = frame['frames']
			#		labels = [l.split('(')[0] for l in labels] 
			#		label_space.update(labels)
			#		cleaned_data.append((tokens, (predicate, labels)))

	return cleaned_data

def process_conll(data, label_space, task):
	cleaned_data = []
	for sent in data:
		tokens = sent['tokens']
		if task == 'ner':
			labels = [label_space[i] for i in sent['ner_tags']]
		elif task == 'chunk':
			labels = [label_space[i] for i in sent['chunk_tags']]
		cleaned_data.append((tokens, labels))
	return cleaned_data

#part of speech tagging on UD with UPOS
def pos_data(eval_on_test=False):
	eval_split = 'dev'
	if eval_on_test: eval_split = 'test'

	#Fill in with top level UD directory that contains the EN treebank within it 
	#(We used the English GUM Treebank, which can be found at https://github.com/UniversalDependencies/UD_English-GUM/tree/master)
	#To use this with other languages, change the lang code "en" to the one used in the filenames of treebank you are interested in
	ud_datapath = None
	prompt_data = load_ud_datasets(ud_datapath, 'en', task='pos', split='train')
	eval_data = load_ud_datasets(ud_datapath, 'en', task='pos', split=eval_split) #1117 dev, 1096 test examples 


	labels = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',\
	'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',\
	'PUNCT', 'SYM', 'X']

	#filter prompt examples that don't have tagged words (i.e., only has "O" tags)
	prompt_data = [(sent, tags) for sent, tags in prompt_data if len(set(tags)) > 1]

	return prompt_data, eval_data, labels

#named entity recognition on CONLL2003 or CONLL2012 
def ner_data(eval_on_test=False):

	eval_split = 'validation'
	if eval_on_test: eval_split = 'test'

	
	labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC',\
	'B-MISC', 'I-MISC']

	from datasets import load_dataset
	prompt_data = load_dataset('conll2003', split='train')
	prompt_data = process_conll(prompt_data, labels, task='ner')
	# 3251 dev, 3454 test examples
	eval_data = load_dataset('conll2003', split=eval_split)
	eval_data = process_conll(eval_data, labels, task='ner')


	#filter prompt examples that don't have tagged words (i.e., only has "O" tags)
	prompt_data = [(sent, tags) for sent, tags in prompt_data if len(set(tags)) > 1]
	return prompt_data, eval_data, labels

#sentence chunking on the CONLL2000 dataset
#https://www.clips.uantwerpen.be/conll2000/chunking/
def chunking_data():
	eval_split = 'test'

	labels = ['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP', 'B-CONJP', 'I-CONJP',\
	'B-INTJ', 'I-INTJ', 'B-LST', 'I-LST', 'B-NP', 'I-NP', 'B-PP', 'I-PP',\
	'B-PRT', 'I-PRT', 'B-SBAR', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-VP', 'I-VP']

	#load ontonotes data
	from datasets import load_dataset
	prompt_data = load_dataset('conll2000', split='train')
	prompt_data = process_conll(prompt_data, labels, task='chunk')
	eval_data = load_dataset('conll2000', split=eval_split) #2,013 test examples
	eval_data = process_conll(eval_data, labels, task='chunk')

	#manually created label mapping since huggingface dataset doesn't provide it

	#filter prompt examples that don't have tagged words (i.e., only has "O" tags)
	prompt_data = [(sent, tags) for sent, tags in prompt_data if len(set(tags)) > 1]

	return prompt_data, eval_data, labels

def ud_f1(gold, pred):
	flat_gold = []
	flat_pred = []
	for x, y in zip(gold, pred):
		flat_gold.extend(x)
		flat_pred.extend(y)

	score = metrics.f1_score(flat_gold, flat_pred, pos_label='B')
	score = score*100

	return score

def ud_pos_accuracy(gold, pred):
	flat_gold = []
	flat_pred = []
	for x, y in zip(gold, pred):
		flat_gold.extend(x)
		flat_pred.extend(y)

	#filter unlabeled marker _
	score = sum([1 for x, y in zip(flat_gold, flat_pred) if (x == y and x != '_')])/len(flat_gold)
	score = score*100
	return score

#EOF