"""
Convert the annotated sentences back into conversations, both generic and courteous.
Author: Hitesh Golchha
"""
import numpy as np 
from utils.general_utils import *
import collections
import csv
import os

def generate_conversations_courteous():
	sent_ids = unpickle('pickle_files/sent_ids.pickle')
	sent_inbounds = unpickle('pickle_files/sent_inbounds.pickle')
	sentences = unpickle('pickle_files/sentences.pickle')
	conversation_company_dict = unpickle('pickle_files/conversation_company_dict.pickle')


	prev_conv_id = 0
	prev_utt_id = 0
	with open('data/coversations_regenerated.csv', 'w+', encoding = 'utf-8') as csvfile:
		fieldnames = ['conv_id', 'utterance_id', 'text', 'inbound', 'company']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames,  lineterminator='\n')
		writer.writeheader()
		data = []
		for i in range(len(sentences)):
			sent = sentences[i]
			conv_id, utt_id, sent_no = sent_ids[i].split('_')  
			conv_id = int(conv_id)
			utt_id = int(utt_id)

			if(conv_id != prev_conv_id):
				prev_utt_id = 0

			if(utt_id == prev_utt_id):
				data[-1]['text'] = data[-1]['text'] + ' ' + sent
			if(str(conv_id) not in conversation_company_dict.keys()):
				conversation_company_dict[str(conv_id)] = '<UNK>'
			else:	
				data.append({'conv_id': conv_id, 'utterance_id': utt_id, 'text': sent, 'inbound': sent_inbounds[i], 'company': conversation_company_dict[str(conv_id)] })
				prev_utt_id = utt_id
				prev_conv_id = conv_id
	
		writer.writerows(data)



def generate_conversations_generic_company_wise():
	conversation_company_dict = unpickle('pickle_files/conversation_company_dict.pickle')
	company_names = [conversation_company_dict[k] for k in conversation_company_dict.keys() ] 
	company_names = list(set(company_names))

	sent_ids = unpickle('pickle_files/sent_ids.pickle')
	sent_inbounds = unpickle('pickle_files/sent_inbounds.pickle')
	sentences = unpickle('pickle_files/sentences.pickle')

	invalid_conversation_ids  = []

	sent_id_dict = {sent_ids[i]: i for i in range(len(sent_ids)) }

	annotated_company_names = []	


	# Convert sentences into generic in nature.
	print('Reading annotated files')
	for annotated_file in os.listdir('data/Annotations_csv'):
		company_name = annotated_file[:-4]
		annotated_company_names.append(company_name)
		print(annotated_file)
		with open('./data/Annotations_csv/{}'.format(annotated_file), 'r+', encoding = 'utf-8') as csv_file:
			line_count = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if line_count == 0:
					print("Column names are {}".format(", ".join(row)))
					line_count += 1
				else:
					line_count += 1
					if company_name != 'AmazonHelp':
						index, conv_id, text, courteous, output = row
					else:
						index, conv_id, text, courteous, eng_language, output = row

					index  = int(index)

					if output!= '':
						sentences[index] = output

					elif courteous == 'y':						
						sentences[index] = ''	

					if company_name == 'AmazonHelp' and eng_language == 'n':
						invalid_conversation_ids.append(conv_id)

	print('Saving conversations which are invalid')					
	invalid_conversation_ids = list(set(invalid_conversation_ids))					
	pickelize(invalid_conversation_ids, 'pickle_files/invalid_conversation_ids.pickle')
	prev_conv_id = 0
	prev_utt_id = 0

	for name in annotated_company_names:
		with open('data/regenerated/companywise_generic/coversations_{}.csv'.format(name), 'w+', encoding = 'utf-8') as csvfile:
			fieldnames = ['conv_id', 'utterance_id', 'text', 'inbound', 'company']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames,  lineterminator='\n')
			writer.writeheader()
			data = []
			for i in range(len(sentences)):

				sent = sentences[i]
				conv_id, utt_id, sent_no = sent_ids[i].split('_')  
				conv_id = int(conv_id)
				utt_id = int(utt_id)

				if(conv_id != prev_conv_id):
					prev_utt_id = 0

				if(str(conv_id) not in conversation_company_dict.keys()):
					conversation_company_dict[str(conv_id)] = '<UNK>'

				if name != conversation_company_dict[str(conv_id)]:
					continue


				if(utt_id == prev_utt_id):
					data[-1]['text'] = data[-1]['text'] + ' ' + sent
				else:	
					data.append({'conv_id': conv_id, 'utterance_id': utt_id, 'text': sent, 'inbound': sent_inbounds[i], 'company': conversation_company_dict[str(conv_id)] })
					prev_utt_id = utt_id
					prev_conv_id = conv_id
		
			writer.writerows(data)

def evaluate_cluster_purity(clusters):
	total = 0
	majority = 0
	for cluster in clusters:
		c = collections.Counter(cluster)
		max_freq = 0
		for i in c.values():
			max_freq = max(max_freq, i) 
		majority = majority + max_freq
		total = total + len(cluster)
		# print(c, c[max(c)], len(cluster))
		if len(c)>2:
			print(c)
	return (majority / total)	




def evaluate_cluster_purity_annotations():

	for filename in os.listdir('data/Annotations_csv'):
		company_name = filename[:-4]
		print(company_name)
		with open('./data/Annotations_csv/{}'.format(filename), 'r+', encoding = 'utf-8') as csv_file:
			clusters = []
			latest_cluster = []
			line_count = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			index_prev = 0	
			for row in csv_reader:
				if line_count == 0:
					line_count += 1
				else:
					line_count += 1
					if company_name != 'AmazonHelp':
						index, conv_id, text, courteous, output = row
					else:
						index, conv_id, text, courteous, eng_language, output = row

					if output!= '':
						courteous = 'n'
					if 'n' in courteous:
						courteous = 'n'
					if 'y' in courteous:
						courteous = 'y'
								
					index = int(index)	
					if index > index_prev:
						latest_cluster.append(courteous)

					else:
						clusters.append(latest_cluster)
						latest_cluster = []
						latest_cluster.append(courteous)
					index_prev = index	

			if latest_cluster != []:
				clusters.append(latest_cluster)
	
			purity = evaluate_cluster_purity(clusters)
			print('The purity of the clusters is: ', purity)




def count_conversations():
	for filename in os.listdir('data/regenerated/companywise_generic'):
		company_name = filename[:-4]
		convs = []
		with open('data/regenerated/companywise_generic/{}'.format(filename), 'r+', encoding = 'utf-8') as csv_file:
			line_count = 0
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if line_count == 0:
					line_count += 1
				else:
					line_count += 1
					conv_id, utterance_id, _, _, _  = row
					convs.append(str(conv_id))
		convs = collections.Counter(convs)
		values = np.asarray(list(convs.values()))
		print(company_name, len(convs.keys()), np.min(values), np.max(values), np.average(values), np.std(values))

def main():
	count_conversations()

if __name__ == '__main__':
	main()