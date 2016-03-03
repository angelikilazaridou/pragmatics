import argparse
import random
from itertools import product,combinations
import copy
import numpy as np
import scipy.io
import h5py
import json
from random import shuffle
import math




def gen_Carina_single_vecs(params):

	data = []

	#load binary vecctors
        vectors = np.loadtxt('../../DATA/visAttCarina/raw/vectors.txt',dtype=int)
        n_concepts = vectors.shape[0]
        vocab_size = vectors.shape[1]
        f = open('../../DATA/visAttCarina/raw/concepts.txt')
        concepts = [a.strip() for a in f.readlines()]
        f.close()
        print "Read ",len(concepts)," carina concepts"


	# create index of images for concepts staring from 1
	index = {}
	line_num = 1
       	with open('../../DATA/visVecs/out_index.txt') as f:
		for line in f.readlines():
			concept = line.strip().split('/')[2]
			#if concept is not
			if concept not in concepts:
				print "Skiping line ",line
				continue
			#add one because of stupid lua
			pos = concepts.index(concept) +1
			if pos not in index:
				index[pos] = []
			index[pos].append(line_num)
			
			line_num +=1
	
	#generate all possible combinations of concepts without repetition 
        for v in list(combinations(range(n_concepts),params['images'])):

                #XOR solution based on binary vector
                c = vectors[v[0]] ^ vectors[v[1]]
                #append data, solution and concepts
                data.append((c,v[0],v[1]))

	imgs = []
        n_imgs = len(data)


        #shuffle data	
	random.seed(100)
        shuffle(data)

	#read single images
	f = h5py.File('../../DATA/visVecs/images_single.h5','r')
	images = np.array(f["images"])
	f.close()
	
	output_h5 = params['output_h5']
	print(images.shape)
        #sava data into hd5
        f = h5py.File(output_h5, "w")
        f.create_dataset("images", (images.shape[1],images.shape[0]), dtype='float32', data=images.transpose()) # space for resized images
        labels = f.create_dataset("labels", (n_imgs,vocab_size),dtype='float32')
        dset2 = f.create_dataset("properties",(n_concepts, vocab_size), dtype='float32')
        for i,el in enumerate(data):
                v = el[0]
		c1 = el[1]
		c2 = el[2]
                # write t
                labels[i] = v
                # create image data
                img = {}
                img['id'] = i+1 #where to find the image 
                img['concepts'] = (c1+1,c2+1) #so that to be able to do analysis
                imgs.append(img)


	for n in range(n_concepts):
                dset2[n] = np.array(vectors[n])
        f.close()

        #generate splits
        if params['zero_shot']>0:
                assign_splits_0shot(imgs,params)
        else:
                assign_splits(imgs,params)

        #save json file
        out = {}
        out['images'] = []
        out['vocab_size'] = vocab_size
        out['game_size']  = params['images']
	out['index'] = index

        for i,img in enumerate(imgs):
                jimg = {}
                jimg['split'] = img['split']
                jimg['id'] = img['id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
                jimg['concepts'] = img['concepts']
                out['images'].append(jimg)
	

	print(len(out['images']))
        json.dump(out, open(params['output_json'], 'w'))


        return data,imgs


#generate data were vectors are real visual vectors
#verlap between TR and TS concepts
def gen_Carina_vis_vecs(params):
        data = []


	#load binary vecctors
        vectors = np.loadtxt('../../DATA/visAttCarina/raw/vectors.txt',dtype=int)
        n_concepts = vectors.shape[0]
        vocab_size = vectors.shape[1]
	f = open('../../DATA/visAttCarina/raw/concepts.txt')
	concepts = [a.strip() for a in f.readlines()]
	f.close()
	print "Read ",len(concepts)," carina concepts"

	#load visual vectors
	vis_vectors = scipy.io.loadmat('../../DATA/visVecs/fc7.mat')['X']
	feat_size = vis_vectors.shape[1]
	f = open('../../DATA/visVecs/fc7labels.txt')
	rows = [a.strip() for a in f.readlines()]
	f.close()
	row2id = {}
	i=0
	for c in rows:
		row2id[c] = i
		i+=1
	print "Read ",len(row2id)," visual concepts"

        output_h5 = params['output_h5']

        #generate all possible combinations of concepts without repetition 
        for v in list(combinations(range(n_concepts),params['images'])):
		c = concepts[v[0]]
		idx = row2id[c]
		a = vis_vectors[idx]
		c = concepts[v[1]]
		idx = row2id[c]
		b  = vis_vectors[idx]
	
                #XOR solution based on binary vector
                c = vectors[v[0]] ^ vectors[v[1]]
                #append data, solution and concepts
                data.append((a,b,c,v[0],v[1]))



        imgs = []
        n_imgs = len(data)

        #shuffle data
        random.seed(100)
        shuffle(data
)
        #sava data into hd5
        f = h5py.File(output_h5, "w")
        dset = f.create_dataset("images", (n_imgs,2,feat_size), dtype='float32') # space for resized images
        labels = f.create_dataset("labels", (n_imgs,vocab_size),dtype='float32')
	dset2 = f.create_dataset("properties",(n_concepts, vocab_size), dtype='float32')
        for i,el in enumerate(data):
                b = el[0]
                c = el[1]
                v = el[2]
		'''
		#convert XOR 0/1 to XOR -1/1
		for j in range(vocab_size):
                        if v[j] == 0:
				v[j] = -1
		'''
                # write t
                dset[i] = np.array([b,c])
                labels[i] = v
                # create image data
                img = {}
                img['id'] = i+1 #where to find the image 
                img['concepts'] = (el[3]+1,el[4]+1) #so that to be able to do analysis
                imgs.append(img)

       
	for n in range(n_concepts):
		dset2[n] = np.array(vectors[n])
	f.close()

        #generate splits
	if params['zero_shot']>0:
		assign_splits_0shot(imgs,params)
	else:
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



#generate data with concept vectors but binary data
#tr and ts concepts overlap
def gen_Carina(params):
	data = []
	
	vectors = np.loadtxt('/home/angeliki/git/pragmatics/DATA/visAttCarina/raw/vectors.txt',dtype=float)
	
	
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
		


#truly 0shot both concepts
def assign_splits_0shot(imgs, params):

	#read in data
	concepts = {}
	c2idx = {}
	
	cat2c = {}

	idx = 1
	with open('../../DATA/visAttCarina/raw/concepts_with_cats.txt') as f:
        	for line in f.readlines():
			c = line.strip().split(' ')[0]
			cat = line.strip().split(' ')[1]
			
			concepts[c] = cat
			
			c2idx[c] = idx
			idx+=1	
	
			if not cat in cat2c:
				cat2c[cat] = []
			cat2c[cat].append(c)


	#keep params.zero_shot apart for test
	ts_concepts = {}
	for cat in cat2c:
		shuffle(cat2c[cat])
		#if there are too few concepts, skip category
		if cat2c < 2:
			continue
		for i in range(int(math.ceil(params['zero_shot'] * len(cat2c[cat])))):
			ts_concepts[c2idx[cat2c[cat][i]]] = cat2c[cat][i]
		
	print "test concepts ids",ts_concepts

	#map them to ids

	num_val = params['num_val']
	val = 0
	for i,img in enumerate(imgs):

		c1,c2 = img['concepts']
		#if test concept
		if (c1 in ts_concepts) or (c2 in ts_concepts):
			if (c1 in ts_concepts) and (c2 in ts_concepts):
				img['split'] = 'test'
			else:
				img['split'] = 'unused'
		else:
			#keep some apart for validation
                	if val < num_val:
                        	img['split'] = 'val'
				val +=1
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
	parser.add_argument('-zero_shot',dest = 'zero_shot', type=float, default=0.1, help="Whether to keep apart for 0shot. If > then this is interpreted as fraction of concepts per category")
	args = parser.parse_args()

	params = vars(args) # convert to ordinary dict


        data, imgs = gen_Carina_vis_vecs(params)
	

	print "Number of asked games:",params['games']
	print "Number of produced games:",len(data)

	print data[0][3:]
