from itertools import combinations, product
import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize


def build_vocab(data_new, params):
	
	count = {}
	for d in data_new:
		for w in d['RE1']:
			if not w in count:
				count[w] = 0
			count[w] +=1
		for w in d['RE2']:
			if not w in count:
	                        count[w] = 0
	                count[w] +=1
	
	vocab = [w for w,n in count.iteritems()]
	
	print 'number of words in vocab would be %d' % (len(vocab), )

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

	vocab_size = len(wtoi)
  	N = len(data)
	label_arrays = []
	ref_arrays = []
	single_label_arrays = []

	for i,d in enumerate(data):
		### wtoi[w]-1 because 1-index vocab
		Li = np.zeros((1,2, vocab_size), dtype='uint32')
		
		#add RE1
		for k,w in enumerate(d['RE1']):
		   	Li[0,0,wtoi[w]-1] = 1
		
		 
		#add RE2
		for k,w in enumerate(d['RE2']):
			Li[0,1,wtoi[w]-1] = 1
   		ref_arrays.append(Li)

		Li2 = np.zeros((1,vocab_size),dtype='uint32')
		#add discr
		l = -1
		for k,w in enumerate(d['discr']):
			Li2[0,wtoi[w]-1] = 1
			l = wtoi[w]
		label_arrays.append(Li2)
		single_label_arrays.append(l)
    	

  	L1 = np.concatenate(label_arrays, axis=0) # put all the labels together
  	L2 = np.concatenate(ref_arrays, axis=0) # put all expressions together
  	print len(single_label_arrays)
  	L3 = np.array(single_label_arrays)
	print 'encoded expressions to array of size ', `L2.shape`,'   ',`L1.shape`,' ',L3.shape
 	return L1,L2,L3

def format_data(images):
	data_new = []

	ims = images.keys()

	#create all combinations of pairs of concepts -> <referent, context>
	for i1, i2 in list(combinations(range(len(ims)),2)):
		if i1==i2:
			continue
		im1 = ims[i1]
		im2 = ims[i2]
			
		d = {}
		d['discr'] = list(set(images[im1][1])-set(images[im2][1]))
		if len(d['discr']) == 0 :
			continue
		d['RE1'] = images[im1][1]
		d['RE2'] = images[im2][1]
		d['bb1'] = im1
		d['bb2'] = im2
		d['bb1_i'] = images[im1][0]
		d['bb2_i'] = images[im2][0]
		d['RE1_original'] = d['RE1']
		d['RE2_original'] = d['RE2']
		d['ratio'] = -1

		data_new.append(d)
	return data_new

def main(params):

	#read images
	images = {}
	i = 1
	with open(params["images_ids"],'r') as f:
		for line in f:
			line = line.strip()
			im = line.split('/')[7]
			images[im] = []
			for attr in im.split('.')[0].split('_'):
				images[im].append(attr)
			images[im] = (i, images[im])
			i= i+1

	#create prettyy jason
	data_new = format_data(images)
  	seed(123) # make reproducible
  	shuffle(data_new) # shuffle the order


	# create the vocab
	vocab = build_vocab(data_new, params)
	itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
	wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

	# throw away some data
	data_new = data_new[0:(params['num_test']+params['num_val']+params['num_train'])]

	# assign the splits
  	assign_splits(data_new, params)
  
	# encode captions in large arrays, ready to ship to hdf5 file
	L1, L2, L3 =  encode_expressions(data_new, params, wtoi)

	# create output h5 file
	N = len(data_new)
 
	f = h5py.File(params['output_h5'], "w")
	f.create_dataset("labels", dtype='uint32', data=L1)
  	f.create_dataset("refs",dtype='uint32',data=L2)
  	f.create_dataset("single_label", dtype='uint32', data=L3)

  	f.close()
  	print 'wrote ', params['output_h5']

	# create output json file
	out = {}
	out['ix_to_word'] = itow # encode the (1-indexed) vocab
	out['refs'] = data_new
	out['vocab_size'] = len(itow)
  
	json.dump(out, open(params['output_json'], 'w'))
	print 'wrote ', params['output_json']

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--num_train',default=100000,type=int,help='traning data')
	parser.add_argument('--images_ids',default="/home/angeliki/git/pragmatics/DATA/toy_data/all_images.txt",help='Image ids to generate data')
	parser.add_argument('--num_val', default=1000, type=int, help='number of images to assign to validation data (for CV etc)')
	parser.add_argument('--output_json', default='data.json', help='output json file')
	parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
	parser.add_argument('--num_test', default=1000, type=int, help='number of test images (to withold until very very end)')
	
	args = parser.parse_args()
  
	params = vars(args) # convert to ordinary dict
	print 'parsed input parameters:'
  	print json.dumps(params, indent = 2)
  	main(params)