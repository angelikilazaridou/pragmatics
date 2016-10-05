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

		--local p_t = nn.Sigmoid()(nn.MulConstant(1)(property_vec))
		local p_t = property_vec
		table.insert(all_prop_vecs,p_t)

	end

	-- in: table game_size x batch_size x embedding_size
	-- out: tensor batch_size x game_size x embedding_size
	local properties_3d = nn.View(game_size, -1):setNumInputDims(1)(nn.JoinTable(2)(all_prop_vecs))

	-- in: tensor batch_size x game_size x embedding_size
	-- out: tensor batch_size * embedding_size x game_size
	local properties_3d_b = nn.View(-1, game_size)(nn.Transpose({2,3})(properties_3d))
	
	--hidden layer for comparison
	local comparison = nn.LinearNB(game_size, hidden_size)(properties_3d_b)
	comparison  =  nn.Sigmoid()(comparison)
	
	--compute discriminativeness
	local result = nn.LinearNB(hidden_size,1)(comparison)
	-- out: batch_size x embedding_size
	result = nn.View(-1,embedding_size)(result)	
	
	result = nn.Sigmoid()(result)

	-- in: batch_size x embedding_size
	-- out: batch_size x vocab_size
	local scores = nn.LinearNB(embedding_size, vocab_size)(result):annotate{name='embeddings_S'}
	
	--take ouput
    	local outputs = {}
	table.insert(outputs, scores)
	

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