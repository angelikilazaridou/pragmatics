require 'nn' 
require 'nngraph'
require 'misc.Peek'
require 'misc.LinearNB'

local player1 = {}
function player1.model()


	local inputs = {}

	for i=1,game_size do
		local image = nn.Identity()() --insert one image at a time
		table.insert(inputs, image)
	end

	--insert predicted feature
	local predicted = nn.Identity()()
	table.insert(inputs,predicted)

	--insert gold
	local gold = nn.Identity()()
	table.insert(inputs, gold)

	

	-- Then convert to 3d -> batch_size x 2 x feat_size
	local properties_3d = nn.View(2,-1):setNumInputDims(1)(nn.JoinTable(2)(all_prop_vecs))
	-- convert to batch_size * feat_size x 2
	local properties_3d_b = nn.View(-1,game_size)(nn.Transpose({2,3})(properties_3d))
	
	--hidden layer for discriminativeness
	local hid = nn.Linear(game_size, hidden_size)(properties_3d_b)
	hid =  nn.Sigmoid()(nn.MulConstant(1)(hid))
	
	--compute discriminativeness
	local discr = nn.Linear(hidden_size,1)(hid)
	if scale_output ==1 then
		discr = nn.Sigmoid()(nn.MulConstant(k)(discr))
	end
	
	--reshaping to batch_size x feat_size
	local result = nn.View(-1,vocab_size)(discr)

	local probs = nn.SoftMax()(result)

    	local outputs = {}
	table.insert(outputs, probs)
	--[[
	--insert all property vectors
	for i=1,game_size do
		table.insert(outputs,all_prop_vecs[i])
	end
	]]--
    	
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
