require 'misc.Peek'
require 'misc.LinearNB'
require 'nngraph'
require 'dp'

local player2 = {}
function player2.model(game_size, feat_size, vocab_size, property_size, embedding_size, hidden_size, dropout, gpu)


	local shareList = {}
	--read in inputs
	local inputs = {}
	local all_vecs = {}
	shareList[1] = {} --share mapping to property space

	for i=1,game_size do
		local image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
		-- dropping out visual dimensions from visual vectors
		local dropped = nn.Dropout(dropout)(image)		
		--map images to some property space
		local property_vec = nn.LinearNB(feat_size, property_size)(dropped)
		table.insert(shareList[1],property_vec)
		-- collecting property vectors
		table.insert(all_vecs,property_vec)

	end


	-- Then convert a table of (game_size x batch_size x property_size) to a  2d matrix of  batch_size x (game_size x property_size)
	-- essentially concatenating vectors within game
	local all_vecs_matrix = nn.JoinTable(2)(all_vecs)

        -- the attribute of P1
        local attribute = nn.Identity()()
        table.insert(inputs, attribute)

	-- embed attribute
	local embedded_attribute = nn.LinearNB(vocab_size, embedding_size)(attribute):annotate{name="embeddings_R"}
	-- putting altogether
	local multimodal = nn.JoinTable(2)({all_vecs_matrix, embedded_attribute})
        
	-- compute interaction between images and attribute vectors 
        local hid = nn.LinearNB(property_size*game_size + embedding_size, hidden_size)(multimodal)
        hid =  nn.Sigmoid()(hid)
	
	-- scores of images in game
	local scores = nn.LinearNB(hidden_size, game_size)(hid)
	scores = nn.Sigmoid()(scores)
	-- probabilities over input
	local probs = nn.SoftMax()(scores)
	
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

return player2
