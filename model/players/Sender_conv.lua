require 'misc.LinearNB'
require 'misc.Peek'
require 'nngraph'
require 'dp'

local player1 = {}
function player1.model(game_size, feat_size, vocab_size, embedding_size, property_size, dropout, gpu)

	local shareList = {}
	shareList[1] = {} --share problem-specific mapping to property space


	local inputs = {}
	local all_prop_vecs = {}
	
	for i=1,game_size do
		local image = nn.Identity()()
		table.insert(inputs, image)
		--map input to some problem specpfic space
		local property_vec = nn.LinearNB(feat_size, property_size)(image):annotate{name='property'}
		
		table.insert(shareList[1],property_vec)
		table.insert(all_prop_vecs,property_vec)
	end

	local filters = 20
	local filters2 = 1
	local dh = 1
	local dpool = 5
	
	-- in: table -> game_size x batch_size x feat_size 
	-- out: tensor -> batch_size x game_size x property_size 
	local matrix_vecs = nn.JoinTable(2)(all_prop_vecs)

	--matrix_vecs = nn.Sigmoid()(matrix_vecs)	
	-- convolve input

	-- in: table -> game_size x batch_size x feat_size
	-- out: batch_size x filters x (property_size-h)+1 x 1
	local conv1 = nn.SpatialConvolution(game_size,filters,1,dh)(nn.Reshape(game_size, property_size, 1)(matrix_vecs))
 	conv1 = nn.Sigmoid()(conv1)
	-- in: batch_size x filters x (property_size-h)+1 x 1
	-- out: batch_size x filters x floor(((property_size-dh)+1)/dpool) x1
	conv1 = nn.SpatialMaxPooling(1,dpool)(conv1)
	conv1 = nn.SpatialConvolution(filters, filters2,1, dh)(conv1)
        conv1 = nn.SpatialMaxPooling(1,dpool)(conv1)
	conv1 = nn.Sigmoid()(conv1)	
	--fully connected to communication layer
	conv1 = nn.View(filters2 * math.floor(((math.floor(((property_size-dh)+1)/dpool)-dh)+1)/dpool))(conv1)
	--conv1 = nn.Dropout(0.5)(conv1)
	local fc = nn.LinearNB(filters2 *math.floor(((math.floor(((property_size-dh)+1)/dpool)-dh)+1)/dpool), embedding_size)(conv1)
	fc = nn.Sigmoid()(fc)

	local scores = nn.LinearNB(embedding_size, vocab_size)(fc):annotate{name='embeddings_S'}

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
