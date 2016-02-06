require 'nn' 
require 'nngraph'
require 'misc.Peek'

local XOR3 = {}
function XOR3.xor(game_size, feat_size, to_share)

	--read in inputs
	local inputs = {}
	for i=1,game_size do
		image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
	end
    
	
	--to accumulate instances
	local all_L1 = {}
	local all_L2 = {}
	for i=1,game_size  do

		local L1 = nn.Mul()(inputs[i])
		table.insert(all_L1,L1)
		
	end
	

	local combined_L1
	--add all images arriving at a mode
	for i=1,game_size do
		if i==1 then
			combined_L1 = all_L1[i]
		else
			combined_L1 = nn.CAddTable()({all_L1[i],combined_L1})

		end
	end

	--apply non-Linearity
        combined_L1 = combined_L1

	--convert to soft-max
	local result = nn.Sigmoid()(combined_L1)

    	local outputs = {}
	table.insert(outputs, result)
    	
	local model = nn.gModule(inputs, outputs)
   
	return model
end

return XOR3
