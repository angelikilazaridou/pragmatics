require 'nn' 
require 'nngraph'
require 'misc.Peek'

local XOR3 = {}
function XOR3.xor(game_size, feat_size, vocab_size, hidden_size, share, gpu, k)


	local shareList = {}
	--read in inputs
	local inputs = {}
	local all_prop_vecs = {}
	shareList[1] = {} --share mapping to property space

	for i=1,game_size do
		local image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
		--map images to some property space
		local property_vec = nn.Linear(feat_size, vocab_size)(image)
		table.insert(shareList[1],property_vec)
		table.insert(all_prop_vecs,property_vec)
	end

	-- convert to batch_size x (feat_size x 2) with JoinTable
	-- OLD: nn.Transpose({2,3})(nn.View(2,-1):setNumInputDims(2)(nn.View(2,-1):setNumInputDims(1)(nn.JoinTable(2)(properties))))
	-- Then convert to 3d -> batch_size x 2 x feat_size
	local properties_3d = nn.View(2,-1):setNumInputDims(2)(nn.View(2,-1):setNumInputDims(1)(nn.JoinTable(2)(all_prop_vecs)))
	--then start selecting per property, map and add into a table
	shareList[2] = {}
	local all_L1 = {}
	for i=1,vocab_size do
		local images = nn.Select(3,i)(properties_3d)
		local L1 = nn.Linear(game_size,hidden_size)(images)
		table.insert(shareList[2],L1)
		table.insert(all_L1,nn.Sigmoid()(nn.MulConstant(1)(L1)))
	end
	
	--take discriminativeness of feature	
	shareList[3] = {}
	local all_L2 = {}
	for i=1,vocab_size do
		local L2 = nn.Linear(hidden_size,1)(all_L1[i])
		table.insert(all_L2,nn.Sigmoid()(nn.MulConstant(k)(L2)))
		table.insert(shareList[3],L2)
	end
	


	--convert to soft-max
	local result = nn.JoinTable(2)(all_L2)

    	local outputs = {}
	table.insert(outputs, result)
    	
	local model = nn.gModule(inputs, outputs)

	if gpu>=0 then model:cuda() end
        -- IMPORTANT! do weight sharing after model is in cuda
	if share==1 then
        	for i = 1,#shareList do
                	local m1 = shareList[i][1].data.module
                	for j = 2,#shareList[i] do
                        	local m2 = shareList[i][j].data.module
                         	m2:share(m1,'weight','bias','gradWeight','gradBias')
                 	end
        	end
	end
   
	return model
end

return XOR3
