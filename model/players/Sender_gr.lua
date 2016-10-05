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

	-- insert one image at a time
	local image = nn.Identity()()
	table.insert(inputs, image)	
	-- drop out dimensions from visual vectors
	local dropped = nn.Dropout(dropout)(image)
	--map images to some property space
	local property_vec = nn.LinearNB(feat_size, property_size)(dropped):annotate{name='property'}
	
 	property_vec = nn.ReLU()(property_vec)	
	
	-- hidden layer for discriminativeness
	local hid = nn.LinearNB(property_size*game_size, embedding_size)(property_vec):annotate{name='fixed'}
	hid = nn.Dropout(0.5)(nn.ReLU()(hid))
	
	-- predicting attributes
	local attributes = nn.LinearNB(embedding_size, vocab_size)(hid):annotate{name='embeddings_S'}
	
	-- probabilities over discriminative
	local probs_attributes = attributes --nn.LogSoftMax()(attributes)
	--take out discriminative 
    	local outputs = {}
	table.insert(outputs, probs_attributes)
	

	local model = nn.gModule(inputs, outputs)
	if gpu >=0 then model:cuda() end

	return model
end

return Sender_gr
