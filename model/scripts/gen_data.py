import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize


def build_vocab(data, params):
	count_thr = -1 #params['word_count_threshold']
	print params['max_len']
	# count up the number of words
	counts = {}
	for i in range(len(data)):
		d = data[i]
		RE1 = d['RE1']
		RE2 = d['RE2']
		t = RE1.split()
		if len(t)>params['max_len']:
			params['max_len'] = len(t)
		t = RE2.split()
		if len(t)>params['max_len']:
                        params['max_len'] = len(t)
		for w in RE1.split()+RE2.split():
			counts[w] = counts.get(w, 0) + 1

	cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
	print 'top words and their counts:'
	print '\n'.join(map(str,cw[:20]))

	# print some stats
	total_words = sum(counts.itervalues())
	print 'total words:', total_words
	bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
	vocab = [w for w,n in counts.iteritems() if n > count_thr]
	bad_count = sum(counts[w] for w in bad_words)
	print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
	print 'number of words in vocab would be %d' % (len(vocab), )
	print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

	return vocab

def assign_splits(data, params):
	num_val = params['num_val']
	num_test = params['num_test']

	for i,d in enumerate(data):
		if i < num_val:
        		d['split'] = 'val'
		elif i < num_val + num_test: 
			d['split'] = 'test'
		else: 
			d['split'] = 'train'

	print 'assigned %d to val, %d to test.' % (num_val, num_test)

def encode_expressions(data, params, wtoi):
	""" 
	encode all referrings into one large array, which will be 1-indexed.
	"""

	max_length = params['max_len']
  	N = len(data)
	label_arrays = []
	label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
	ref_arrays = []

	for i,d in enumerate(data):
		Li = np.zeros((1,2, max_length), dtype='uint32')
		#add RE1
		for k,w in enumerate(d['RE1'].split()):
          		Li[0,0,k] = wtoi[w]
		#add RE2
		for k,w in enumerate(d['RE2'].split()):
                        Li[0,1,k] = wtoi[w]
		ref_arrays.append(Li)

		Li2 = np.zeros((1,max_length),dtype='uint32')
		#add discr
                for k,w in enumerate(d['discr'].split()):
                        Li2[0,k] = wtoi[w]
    		label_arrays.append(Li2)

  	L1 = np.concatenate(label_arrays, axis=0) # put all the labels together
  	L2 = np.concatenate(ref_arrays, axis=0) # put all expressions together
	print 'encoded expressions to array of size ', `L2.shape`,'   ',`L1.shape`
 	return L1,L2

def format_data(data, images_index):
	data_new = []	
	for  r,bb1, RE1, bb2, RE2, RE1_original, RE2_original, spatial_1, spatial_2,discr,wo in data:
		d = {}
		d['discr'] = discr
		d['RE1'] = RE1
		d['RE2'] = RE2
		d['bb1'] = bb1
		d['bb2'] = bb2
		d['bb1_i'] = 0
		d['bb2_i'] = 0
		d['RE1_original'] = RE1_original
		d['RE2_original'] = RE2_original
		d['ratio'] = r
		if not bb1 in images_index or not bb2 in images_index:
			continue
		d['bb1_i'] = images_index[bb1]
		d['bb2_i'] = images_index[bb2]

		data_new.append(d)
	return data_new

def main(params):

	data = json.load(open(params['input_json'], 'r'))
	#read images
	images_index = {}
	i = 1
	with open(params["images"],'r') as f:
		for line in f:
			line = line.strip()
			im = line.split('.')[0]
			images_index[im] = i
			i+=1

	#create prerry jason
	data_new = format_data(data, images_index)
  	seed(123) # make reproducible
  	shuffle(data_new) # shuffle the order


	# create the vocab
	vocab = build_vocab(data_new, params)
	itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
	wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

	# assign the splits
  	assign_splits(data_new, params)
  
	# encode captions in large arrays, ready to ship to hdf5 file
	L1, L2 =  encode_expressions(data_new, params, wtoi)

	# create output h5 file
	N = len(data_new)
 
	f = h5py.File(params['output_h5'], "w")
	f.create_dataset("labels", dtype='uint32', data=L1)
  	f.create_dataset("refs",dtype='uint32',data=L2)

  	f.close()
  	print 'wrote ', params['output_h5']

	# create output json file
	out = {}
	out['ix_to_word'] = itow # encode the (1-indexed) vocab
	out['refs'] = data_new
  
	json.dump(out, open(params['output_json'], 'w'))
	print 'wrote ', params['output_json']

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# input json
	#distractors.json
	parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
	parser.add_argument('--num_val', default=1000, type=int, help='number of images to assign to validation data (for CV etc)')
	parser.add_argument('--output_json', default='data.json', help='output json file')
	parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
	parser.add_argument('--num_test', default=1000, type=int, help='number of test images (to withold until very very end)')

	args = parser.parse_args()
  
	params = vars(args) # convert to ordinary dict
	params['images'] = "/home/angeliki/sas_adam/matconvnet-1.0-beta18/ALL_REFERIT.txt"
	params['max_len']= -1 
	print 'parsed input parameters:'
  	print json.dumps(params, indent = 2)
  	main(params)
