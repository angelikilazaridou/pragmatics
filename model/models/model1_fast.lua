require 'nn' 
require 'nngraph'
require 'misc.Peek'
require 'misc.LinearNB'

local model_fast = {}
function model_fast.model(game_size, feat_size, vocab_size, hidden_size, share, gpu, k)


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
		local p_t = property_vec
		table.insert(all_prop_vecs,p_t)
	end

	-- Then convert to 3d -> batch_size x 2 x feat_size
	local properties_3d = nn.View(2,-1):setNumInputDims(1)(nn.JoinTable(2)(all_prop_vecs))
	-- convert to batch_size * feat_size x 2
	local properties_3d_b = nn.View(-1,game_size)(nn.Transpose({2,3})(properties_3d))
	
	--hidden layer for discriminativeness
	local hid = nn.Linear(game_size, hidden_size)(properties_3d_b)
	hid =  nn.Sigmoid()(nn.MulConstant(1)(hid))
	
	--compute discriminativeness
	local discr = nn.Linear(hidden_size,1)(hid)
	-- discr = nn.Sigmoid(nn.MulConstant(k)(discr))
	
	--reshaping to batch_size x feat_size
	local result = nn.View(-1,vocab_size)(discr)

    	local outputs = {}
	table.insert(outputs, result)
	--insert all property vectors
	for i=1,game_size do
		table.insert(outputs,all_prop_vecs[i])
	end
    	
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

return model_fast
