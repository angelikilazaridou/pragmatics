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
	local property_vec = nn.LinearNB(feat_size, vocab_size)(dropped):annotate{name='property'}
	
 	--property_vec = nn.Sigmoid()(property_vec)	
	
	--take out discriminative 
    	local outputs = {}
	table.insert(outputs, property_vec)
	

	local model = nn.gModule(inputs, outputs)
	if gpu >=0 then model:cuda() end

	return model
end

return Sender_gr
