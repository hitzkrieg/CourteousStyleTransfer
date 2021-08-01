import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import spacy
import csv
from tqdm import tqdm 
import numpy as np 
import pickle

nlp = spacy.load('en_core_web_lg')


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def pickelize(obj, filename):
	with open(filename, 'wb') as fp:
		pickle.dump(obj,fp)

def unpickle(filename):
	with open(filename, 'rb') as fp:
		return pickle.load(fp)


def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  # for set_name in ['train', 'val', 'test']:
  for set_name in ['train']:

    print ("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

def verify_files():
	print("Loading generic conversations")
	data = []
	company_names = []
	vocab = []

	for filename in os.listdir('data/regenerated/companywise_generic'):
		company_name = filename[:-4]
		print(company_name)
		company_names.append(company_name)
		with open('data/regenerated/companywise_generic/{}'.format(filename), 'r+', encoding = 'utf-8') as csv_file:
			line_count1 = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if line_count1 == 0:
					line_count1 += 1
				else:
					line_count1 += 1
	print("*****")
	# Now load the courteous conversations and merge with the generic data
	print("Loading courteous conversations")

	for company_name in company_names:
		print(company_name)
		convs = []
		with open('data/regenerated/companywise_generic/{}'.format(filename), 'r+', encoding = 'utf-8') as csv_file:
			line_count2 = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if line_count2 == 0:
					line_count2 += 1
				else:
					line_count2 += 1

	assert line_count1 == line_count2
	data = []
	company_names = []
	vocab = []



def read_conversations_and_convert_into_training_data():
	"""
	"""

	# First load the generic conversations
	print("Loading generic conversations")
	data = []
	company_names = []
	vocab = []

	for filename in os.listdir('data/regenerated/companywise_generic'):
		company_name = filename[:-4]
		print(company_name)
		company_names.append(company_name)
		with open('data/regenerated/companywise_generic/{}'.format(filename), 'r+', encoding = 'utf-8') as csv_file:
			line_count1 = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if line_count1 == 0:
					line_count1 += 1
				else:
					line_count1 += 1
					conv_id, utterance_id, text, inbound, company  = row
					doc = nlp(text)
					text = ' '.join(token.text for token in doc)
					row[2] = text
					data.append(row)
	print("*****")
	# Now load the courteous conversations and merge with the generic data
	print("Loading courteous conversations")

	for company_name in company_names:
		print(company_name)
		convs = []
		with open('data/regenerated/companywise_generic/{}'.format(filename), 'r+', encoding = 'utf-8') as csv_file:
			line_count2 = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if line_count2 == 0:
					line_count2 += 1
				else:
					line_count2 += 1
					conv_id, utterance_id, text, inbound, company  = row
					doc = nlp(text)
					vocab = vocab + [token.text for token in doc]
					text_sents = [' '.join([token.text for token in sent]) for sent in doc.sents]
					text = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in text_sents])
					row[2] = text
					data[line_count2 - 2].append(row[2])

	assert line_count1 == line_count2
	pickelize(data, 'pickle_files/model_input_data_py3.pickle') 

	# Create data into training format
	print("Creating data")
	train = []
	current_conv = -1
	prev_utterances = []

	for i in range(len(data)):
		conv_id, utterance_id, text_generic, inbound, company, text_courteous  = row
		if conv_id != current_conv:
			current_conv = conv_id
			prev_utterances = []
		if inbound == 'FALSE':
			prev_utt_reduced = prev_utterances[-3:]
			tf_example = example_pb2.Example()
			tf_example.features.feature['generic'].bytes_list.value.extend([generic])
			tf_example.features.feature['courteous'].bytes_list.value.extend([courteous])
			for j in range(len(prev_utt_reduced)):
				tf_example.features.feature['history{}'.format(j)].bytes_list.value.extend([prev_utt_reduced[j]])
			tf_example_str = tf_example.SerializeToString()
			train.append(tf_example_str)

		prev_utterances.append(text_courteous)

	# Writing to the binary format
	print("Writing to binary format")

	with open('finished_files/train.bin', 'wb') as writer:
		for tf_example_str in train:
			str_len = len(tf_example_str)
			writer.write(struct.pack('q', str_len))
			writer.write(struct.pack('%ds' % str_len, tf_example_str))

	# Write the vocab file
	print("Writing the vocab file")
	vocab_counter = collections.Counter(vocab)

	with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
	  for word, count in vocab_counter.most_common(VOCAB_SIZE):
	    writer.write(word + ' ' + str(count) + '\n')


	# Chunkize
	print("Chunking binarized file")
	chunk_all()


def main():
	verify_files()
	read_conversations_and_convert_into_training_data()


if __name__ == '__main__':
	main()