require 'misc.LinearNB'
require 'misc.Peek'
require 'nngraph'
require 'dp'

local player1 = {}
function player1.model(game_size, feat_size, vocab_size, embedding_size, hidden_size, dropout, gpu)

	local shareList = {}
	--read in inputs
	local inputs = {}
	local all_prop_vecs = {}
	shareList[1] = {} --share mapping to property space

	for i=1,game_size do
		local image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
		local dropped = nn.Dropout(dropout)(image)
		--map images to some property space
		local property_vec = nn.LinearNB(feat_size, embedding_size)(dropped):annotate{name='property'}
		table.insert(shareList[1],property_vec)

		local p_t = property_vec
		
		table.insert(all_prop_vecs,p_t)

	end


	-- Then convert to 3d -> batch_size x 2 x property_size
	local properties_3d = nn.View(game_size, -1):setNumInputDims(1)(nn.JoinTable(2)(all_prop_vecs))
	-- convert to batch_size * property_size x 2
	local properties_3d_b = nn.View(-1, game_size)(nn.Transpose({2,3})(properties_3d))
	
	--hidden layer for discriminativeness
	local hid = nn.LinearNB(game_size, hidden_size)(properties_3d_b)
	hid  =  nn.Sigmoid()(hid)
	
	--compute discriminativeness
	local discr = nn.LinearNB(hidden_size,1)(hid)
	--reshaping to batch_size x feat_size
	local result = nn.View(-1,embedding_size)(discr)
	result = nn.SoftMax()(result)
	-- probabilities over discriminative
	local embeddings = nn.LinearNB(embedding_size, vocab_size)(result):annotate{name='embeddings_S'}
	local probs = nn.SoftMax()(embeddings)
	--take out discriminative 
    	local outputs = {}
	table.insert(outputs, probs)
	

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
