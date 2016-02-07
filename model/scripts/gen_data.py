import argparse
import random
from itertools import product,combinations
import copy
import numpy as np
import h5py
import json
from random import shuffle



def gen_Carina(params):
	data = []
	
	vectors = np.loadtxt('/home/angeliki/Documents/projects/git/PRAGMATICS/DATA/visAttCarina/raw/vectors.txt',dtype=float)
	
	
	n_concepts = vectors.shape[0]
	vocab_size = vectors.shape[1]
	
	output_h5 = params['output_h5']	

	#generate all possible combinations of concepts without repetition 
	for v in list(combinations(range(n_concepts),params['images'])):	
		a = copy.copy(vectors[v[0]])
		b = copy.copy(vectors[v[1]])
		#XOR solution is handcoded layer
		#c = a^b
		c = np.zeros(vocab_size)
		#append data, solution and concepts
		data.append((a,b,c,v[0],v[1]))


	imgs = []
	n_imgs = len(data)

	#shuffle data
	shuffle(data)

	#sava data into hd5
	f = h5py.File(output_h5, "w")
	dset = f.create_dataset("images", (n_imgs,2,vocab_size), dtype='float32') # space for resized images
	labels = f.create_dataset("labels", (n_imgs,vocab_size),dtype='float32')
  	for i,el in enumerate(data):
    		b = el[0]
		c = el[1]
		v = el[2]
		#convert XOR 0/1 to XOR -1/1
		for j in range(vocab_size):
			if b[j] == 1:
				b[j] = random.uniform(0,1)
				if c[j] == 1:
					c[j] = random.uniform(0,1)
					v[j] = 1
				else:
					c[j] = random.uniform(-1,0)
					v[j] = -1
			else:
				b[j] = random.uniform(-1,0)
				if c[j] == 1:
                                        c[j] = random.uniform(0,1)
                                        v[j] = -1
                                else:
                                        c[j] = random.uniform(-1,0)
                                        v[j] = 1
		print b[:3],c[:3],v[:3]
		# write t
		dset[i] = np.array([b,c])
		labels[i] = v 
		# create image data
		img = {}
                img['id'] = i+1 #where to find the image 
		img['concepts'] = (el[3]+1,el[4]+1) #so that to be able to do analysis
		imgs.append(img)

	f.close()

	#generate split
	assign_splits(imgs,params)

	#save json file
	out = {}
  	out['images'] = []
  	out['vocab_size'] = vocab_size
	out['game_size']  = params['images']
	
	for i,img in enumerate(imgs):
    		jimg = {}
    		jimg['split'] = img['split']
    		jimg['id'] = img['id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
		jimg['concepts'] = img['concepts']
    		out['images'].append(jimg)
		
  
  	json.dump(out, open(params['output_json'], 'w'))


	return data,imgs


def gen_bla(games, images, vocab_size):
	
	print games, images, vocab_size

	data = []

	for g in range(games):
                #create ALL but 1 vectors
		Vs = [[random.randint(0,1) for i in range(vocab_size)] for j in range(images-1)]
		
		#find discriminative features
		discriminative = [1 for i in range(vocab_size)]
		prev = Vs[0]
		for v in Vs:
		 	discriminative = [int(a==b)*c for a,b,c in zip(prev,v,discriminative)]
			prev = v
		
		#find positions of all discriminative features
		indices = [i for i, x in enumerate(discriminative) if x == 1]
		if not indices:
			continue
		#get randomly one discriminative feature 
		feature = indices[random.randint(0,len(indices)-1)]
		#the value should be the oposite of current values
		value = 1-prev[feature]
		#construct picked image
		v = [random.randint(0,1) for i in range(vocab_size)]
		v[feature]  = value
		
		#insert picked image in list of other images
		pos = random.randint(0,len(Vs))
		Vs.insert(pos,v)
		
		#insert game in data
		data.append((Vs,pos,feature))
		
		
	return data


def assign_splits(imgs, params):
	num_val = params['num_val']
  	num_test = params['num_test']

  	for i,img in enumerate(imgs):
		
      		if i < num_val:
        		img['split'] = 'val'
      		elif i < num_val + num_test: 
        		img['split'] = 'test'
      		else: 
        		img['split'] = 'train'
		

def gen_XOR(params):
	data = []
	
	vocab_size = params['vocab_size']
	output_h5 = params['output_h5']	

	#generate all possible combinations of 0/1 with length vocab_size-1
	for v in list(product([0.0, 1.0], repeat=vocab_size-1)):	
		v = list(v)
		#inseet discriminative positions in all possible plaves
		for i in range(vocab_size):
			b = copy.copy(v)
			b.insert(i,0.0)
			c = copy.copy(v)
			c.insert(i,1.0)	
			data.append((b,c,i))


	imgs = []
	n_imgs = len(data)

	#shuffle data
	shuffle(data)

	#sava data into hd5
	f = h5py.File(output_h5, "w")
	dset = f.create_dataset("images", (n_imgs,2,vocab_size), dtype='float32') # space for resized images
	labels = f.create_dataset("labels", (n_imgs,1),dtype='i8')
  	for i,el in enumerate(data):
		print i
    		b = el[0]
		c = el[1]
		v = el[2]
		# write to h5
		dset[i] = np.array([b,c])
		labels[i] = v +1 
		# create image data
		img = {}
		img['discr_feat'] = v+1 #the position of the gold feature
                img['id'] = i+1 #where to find the image 
		imgs.append(img)

	f.close()

	#generate split
	assign_splits(imgs,params)

	#save json file
	out = {}
  	out['images'] = []
  	out['vocab_size'] = params['vocab_size']
	out['game_size']  = 2
	
	for i,img in enumerate(imgs):
    		jimg = {}
    		jimg['split'] = img['split']
    		jimg['id'] = img['id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
		jimg['discr_feat'] = img['discr_feat']
    		out['images'].append(jimg)
  
  	json.dump(out, open(params['output_json'], 'w'))


	return data,imgs

def print_games(data):
	for i in range(len(data)):
		print "Game ",i
		Vs = data[i][0]
		pos = data[i][1]
		feature = data[i][2]
		for j in range(len(Vs)):
			if j!=pos:
				print Vs[j]






if __name__ ==  "__main__":

	parser = argparse.ArgumentParser(description='Bla')
	#game parameters
	parser.add_argument('-games', dest='games', type=int, default=1000,  help='Number of games')
	parser.add_argument('-images', dest='images', type=int, default=2, help='Number of images per game')
	parser.add_argument('-vocab_size',dest='vocab_size',default=10, type=int, help='Size of vocabulary')
	#saving stuff
	parser.add_argument('-output_h5',dest='output_h5',default='data.h5',help='Where to save image features')
	parser.add_argument('-output_json',dest='output_json',default='data.json',help='Where to save image data')
	#training parameters
	parser.add_argument('-num_val',dest ='num_val',type=int,default=500,help="Number of validation elements")
	parser.add_argument('-num_test',dest ='num_test',type=int,default=1000,help="Number of test elements")
	args = parser.parse_args()

	params = vars(args) # convert to ordinary dict


        data, imgs = gen_Carina(params)
	

	print "Number of asked games:",params['games']
	print "Number of produced games:",len(data)

	print data[0][3:]
