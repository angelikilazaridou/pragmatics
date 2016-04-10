require 'misc.Peek'
require 'misc.LinearNB'
require 'nngraph'
require 'dp'

local player2 = {}
function player2.model(game_size, feat_size, vocab_size, hidden_size, gpu)


	local shareList = {}
	--read in inputs
	local inputs = {}
	local all_prop_vecs = {}
	shareList[1] = {} --share mapping to property space

	for i=1,game_size do
		local image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
		
		--map images to some property space
		local property_vec = nn.LinearNB(feat_size, vocab_size)(image)
		table.insert(shareList[1],property_vec)

		local p_t = property_vec
		table.insert(all_prop_vecs,p_t)

	end


	-- Then convert to 3d -> batch_size x 2 x vocab_size
	local properties_3d = nn.View(2,-1):setNumInputDims(1)(nn.JoinTable(2)(all_prop_vecs))

        -- the attribute of P1
        local property = nn.Identity()()
        table.insert(inputs, property)
	-- convert to 3d
	local property_3d = nn.View(1,-1):setNumInputDims(1)(property)
        
	-- batch_size x 2
	local selection = nn.MM(false,true)({properties_3d, property_3d})
	local result = nn.View(-1,2)(selection)
	
	-- probabilities over input
	local probs = nn.SoftMax()(result)
	
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
