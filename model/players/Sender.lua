require 'misc.LinearNB'
require 'misc.Peek'
require 'nngraph'
require 'dp'

local Sender = {}
function Sender.model(game_size, feat_size, vocab_size, embedding_size, property_size, dropout, gpu)


	local shareList = {}
	--read in inputs
	local inputs = {}
	local all_vecs = {}
	shareList[1] = {} --share mapping to property space

	for i=1,game_size do
		-- insert one image at a time
		local image = nn.Identity()()
		table.insert(inputs, image)	
		-- drop out dimensions from visual vectors
		local dropped = nn.Dropout(dropout)(image)
		--map images to some property space
		local property_vec = nn.LinearNB(feat_size, property_size)(dropped):annotate{name='property'}
		
		-- sharing property mapping
		table.insert(shareList[1],property_vec)

		local non_linear = nn.ReLU()(property_vec)
		table.insert(all_vecs,non_linear)


	end


	-- Convert table of game_size x batch_size x property_size to 3d tensor of batch_size x (game_size x property_size) 
	-- essentially concatenating images in the game.
	local all_vecs_matrix = nn.JoinTable(2)(all_vecs)
	
	-- hidden layer for discriminativeness
	local hid = nn.LinearNB(property_size*game_size, embedding_size)(all_vecs_matrix):annotate{name='fixed'}
	hid =  nn.Dropout(0.5)(nn.ReLU()(hid))
	
	
	-- predicting attributes
	local attributes = nn.LinearNB(embedding_size, vocab_size)(hid):annotate{name='embeddings_S'}

		
	-- probabilities over discriminative
	local probs_attributes = attributes --nn.SoftMax()(attributes)
	--take out discriminative 
    	local outputs = {}
	table.insert(outputs, probs_attributes)
	

	local model = nn.gModule(inputs, outputs)

	if gpu>=0 then model:cuda() end
        -- IMPORTANT! do weight sharing after model is in cuda
       	for i = 1,#shareList do
               	local m1 = shareList[i][1].data.module
               	for j = 2,#shareList[i] do
                       	local m2 = shareList[i][j].data.module
                       	m2:share(m1,'weight','bias','gradWeight','gradBias')
               	end
       	end
   
	return model
end

return Sender
