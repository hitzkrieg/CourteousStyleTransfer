import numpy  as np 
import nltk
import re
from utils.general_utils import pickelize, unpickle
import csv
from tqdm import tqdm
import spacy 
nlp = spacy.load('en_core_web_lg')




def preprocess_sentence(sent):
	"""
	Replace links, twitter handles and convert into lowercase.
	"""
	sent = re.sub(r'http\S+', '<url>', sent.lower())
	sent = re.sub(r'@\S+', '<twitter_handle>', sent)
	return sent 



def extract_sentences():
	with open('./data/conversations_selected.csv', 'r+', encoding = 'utf-8') as csv_file:
		line_count = 0

		csv_reader = csv.reader(csv_file, delimiter=',')
		sentences = []
		sent_ids = []
		sent_inbounds  = []
		conv_id_prev = 0	
		utt_count = 0	
		for row in tqdm(csv_reader):
			if line_count == 0:
				print("Column names are {}".format(", ".join(row)))
				line_count += 1
			else:
				line_count += 1
				conv_id, author_id, inbound, text, tweet_id, created_at, response_tweet_id, in_response_to_tweet_id = row
				if conv_id_prev!=  conv_id:
					utt_count = 1
				else:
					utt_count+=1	
				doc = nlp(text)
				tweet_sents = [sent.text for sent in doc.sents]
				if line_count <10:
					print(tweet_sents)
				# tweet_sents = nltk.sent_tokenize(text)
				tweet_sents = list(map(preprocess_sentence, tweet_sents))
				for i, sent in enumerate(tweet_sents):
					sent_ids.append(str(conv_id)+ '_' + str(utt_count) + '_' + str(i+1))
					sent_inbounds.append(inbound)
				sentences = sentences + tweet_sents
				conv_id_prev = conv_id

		pickelize(sentences, 'pickle_files/sentences_spacy.pickle')
		pickelize(sent_ids, 'pickle_files/sent_ids_spacy.pickle')
		pickelize(sent_inbounds, 'pickle_files/sent_inbounds_spacy.pickle')
	
		

def main():
	extract_sentences()


if __name__ == '__main__':
	main()









