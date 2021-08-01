# -*- coding: utf-8 -*-

""" Use DeepMoji to encode sentences into emotional feature vectors.
"""
from __future__ import print_function, division
import sys
sys.path.append('resources/DeepMoji-master')
sys.path.append('resources/DeepMoji-master/examples')
import example_helper
import json
import csv
import numpy as np
from tqdm import tqdm 
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import pickle
import os
import time
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def unpickle(filename):
	with open(filename, 'rb') as fp:
		return pickle.load(fp)

def pickelize(obj, filename):
	with open(filename, 'wb') as fp:
		pickle.dump(obj, fp)


# TEST_SENTENCES = [u'I love mom\'s cooking',
#                   u'I love how you never reply back..',
#                   u'I love cruising with my homies',
#                   u'I love messing with yo mind!!',
#                   u'I love you and now you\'re just gone..',
#                   u'This is shit',
#                   u'This is the shit']

SENTENCES = unpickle('pickle_files/sentences_2.pickle')
maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)

for i in range:(39):
	print("**********")
	print("Batch number ", i+1)
	#time1_start = time.time()
	tokenized, _, _ = st.tokenize_sentences(SENTENCES[i*100000, (i+1)*100000 ])
	#time1_end = time.time()
	#print("Seconds for tokenizing sentences: ", time1_end - time1_start)
	print('Loading model from {}.'.format(PRETRAINED_PATH))
	model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
	model.summary()

	print('Encoding texts..')
	#time2_start = time.time()
	encoding = model.predict(tokenized, verbose=1)
	#time2_end = time.time()
	#print("Seconds for predicting: ", time2_end - time2_start)
	print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[0]))
	print(encoding[0, :5])

	pickelize(encoding, 'pickle_files/emotional_encodings{}_2.pickle'.format(i+1))

	print("No of encodings and the shape : ", len(encoding), encoding.shape)
	keras.backend.clear_session()

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.
