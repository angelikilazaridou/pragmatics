require 'nn' 
require 'nngraph'
require 'misc.Peek'

local XOR3 = {}
function XOR3.xor(game_size, feat_size, to_share)

	--read in inputs
	local inputs = {}
	for i=1,feat_size do
		image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
	end
    
	local shareList = {}
        shareList[1] = {}
	
	--to accumulate instances
	local all_L1 = {}
	for i=1,feat_size  do
		local L1 = nn.Linear(2,20)(inputs[i])
		table.insert(all_L1,nn.Tanh()(L1))
		table.insert(shareList[1],L1)
		
	end
	
	shareList[2] = {}
	local all_L2 = {}
	for i=1,feat_size do
		local L2 = nn.Linear(20,1)(all_L1[i])
		table.insert(all_L2,L2)
		table.insert(shareList[2],L2)
	end
	


	--convert to soft-max
	local result = nn.JoinTable(2)(all_L2)

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

return XOR3
