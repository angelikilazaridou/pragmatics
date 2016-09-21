require 'misc.LinearNB'
require 'misc.Peek'
require 'nngraph'
require 'dp'

local player1 = {}
function player1.model(game_size, feat_size, vocab_size, property_size, hidden_size, dropout, gpu)


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
		
		table.insert(all_vecs,property_vec)

		-- sharing property mapping
		table.insert(shareList[1],property_vec)
		
	end


	-- Convert table of game_size x batch_size x property_size to 3d tensor of batch_size x (game_size x property_size) 
	-- essentially concatenating images in the game.
	local all_vecs_matrix = nn.JoinTable(2)(all_vecs)
	
	-- hidden layer for discriminativeness
	local hid = nn.LinearNB(property_size*game_size, hidden_size)(all_vecs_matrix)
	hid =  nn.Sigmoid()(hid)
	
	
	-- predicting attributes
	local attributes = nn.LinearNB(hidden_size, vocab_size)(hid)
	
	-- probabilities over discriminative
	local probs_attributes = nn.SoftMax()(attributes)
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

return player1
