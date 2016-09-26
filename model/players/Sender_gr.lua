require 'misc.LinearNB'
require 'misc.Peek'
require 'nngraph'
require 'dp'

local Sender_gr = {}
function Sender_gr.model(game_size, feat_size, vocab_size, property_size, embedding_size, dropout, gpu)


	local shareList = {}
	--read in inputs
	local inputs = {}
	local all_vecs = {}
	shareList[1] = {} --share mapping to property space

	-- insert one image at a time
	local image = nn.Identity()()
	table.insert(inputs, image)	
	-- drop out dimensions from visual vectors
	local dropped = nn.Dropout(dropout)(image)
	--map images to some property space
	local property_vec = nn.LinearNB(feat_size, property_size)(dropped):annotate{name='property'}
	
	-- sharing property mapping
	table.insert(shareList[1],property_vec)
		


	-- Convert table of game_size x batch_size x property_size to 3d tensor of batch_size x (game_size x property_size) 
	-- essentially concatenating images in the game.
	local all_vecs_matrix = property_vec
	
	-- hidden layer for discriminativeness
	local hid = nn.LinearNB(property_size*game_size, embedding_size)(all_vecs_matrix)
	hid =  nn.Sigmoid()(hid)
	
	
	-- predicting attributes
	local attributes = nn.LinearNB(embedding_size, vocab_size)(hid):annotate{name='embeddings_S'}
	
	-- probabilities over discriminative
	local probs_attributes = nn.SoftMax()(attributes)
	--take out discriminative 
    	local outputs = {}
	table.insert(outputs, probs_attributes)
	

	local model = nn.gModule(inputs, outputs)
	return model
end

return Sender_gr
