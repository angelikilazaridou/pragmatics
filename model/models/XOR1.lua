require 'nn' 
require 'nngraph'
require 'misc.Peek'

local XOR1 = {}
function XOR1.xor(game_size, feat_size, to_share)

	--read in inputs
	local inputs = {}
	for i=1,game_size do
		image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
	end
    
	--share list, just in case
    	local shareList = {}
    	shareList[1] = {}
	--shareList[2] = {}
	
	--to accumulate instances
	local all_L1 = {}
	local all_L2 = {}
	for i=1,game_size  do

		local L1 = nn.Mul()(inputs[i])
		table.insert(all_L1,L1)
		--share across images
		table.insert(shareList[1],L1)
		
		
		local L2 = nn.Mul()(inputs[i])
		table.insert(all_L2,L2)
		--share across images
		table.insert(shareList[1],L2)
	end
	

	local combined_L1, combined_L2
	--add all images arriving at a mode
	for i=1,game_size do
		if i==1 then
			combined_L1 = all_L1[i]
			combined_L2 = all_L2[i]
		else
			combined_L1 = nn.CAddTable()({all_L1[i],combined_L1})
			combined_L2 = nn.CAddTable()({all_L2[i],combined_L2})

		end
	end

	--apply non-Linearity
        combined_L1 = nn.Sigmoid()(combined_L1)
	combined_L2 = nn.Sigmoid()(combined_L2)

	
	--add layers
	local combination = nn.CAddTable()({combined_L1, combined_L2})
	

	--convert to soft-max
	local result = nn.LogSoftMax()(combination)

    	local outputs = {}
	table.insert(outputs, result)
    	
	local model = nn.gModule(inputs, outputs)
   
	--model:cuda()
    	-- IMPORTANT! do weight sharing after model is in cuda
	if to_share == 1 then
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

return XOR1
