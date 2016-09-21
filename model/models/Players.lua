require 'dpnn'
require 'nn'

local player1 = require 'models.player1'
local player2 = require 'models.player2'

local players, parent =  torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	-- params
	self.batch_size = opt.batch_size
	self.vocab_size = opt.vocab_size
	self.game_size = opt.game_size
	self.embedding_size = opt.embedding_size
	--defining the two players
	self.player1 = player1.model(opt.game_size, opt.feat_size, opt.vocab_size, opt.property_size, opt.hidden_size, opt.dropout, opt.gpuid) 
	self.player2 = player2.model(opt.game_size, opt.feat_size, opt.vocab_size, opt.property_size, opt.embedding_size, opt.hidden_size, opt.dropout, opt.gpuid)
	
	if opt.gpuid == 0 then
		-- categorical for selection of feature
		self.feature_selection = nn.ReinforceCategorical(true):cuda()
		self.image_selection = nn.ReinforceCategorical(true):cuda()
		-- baseline 
	        self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1)):cuda()
	else
		self.feature_selection = nn.ReinforceCategorical(true)
		self.image_selection = nn.ReinforceCategorical(true)
		self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1))
	end


end

--called from game:forward()
function players:updateOutput(input)

	-- input: 	
	local im1a = input[1]
	local im1b = input[2]
	local im2a = input[3]
	local im2b = input[4]
	local temp = input[5]

	--player 1 receives 2 images --
	-- does a forward and gives back 1 action -> 1 feature
	self.probs = self.player1:forward({im1a, im1b})

	--sample a feature
	self.sampled_feat = self.feature_selection:forward({self.probs, temp})
	--[[to_print = torch.random(1000)
	if to_print%1000 == 0 then
		print(self.sampled_feat)
	end--]]

	-- player 2 receives 2 refs and 1 feature and predicts L or R
	self.prediction = self.player2:forward({im2a, im2b, self.sampled_feat})
	-- sample image
	self.sampled_image = self.image_selection:forward({self.prediction, temp})
	-- baseline
	local baseline = self.baseline:forward(torch.CudaTensor(self.batch_size,1))

	local outputs = {}
	table.insert(outputs, self.sampled_image)
	table.insert(outputs, baseline)
	table.insert(outputs, self.sampled_feat)

	return outputs
end


--called from game:backward()
function players:updateGradInput(input, gradOutput)

	-- input:       
        local im1a = input[1]
        local im1b = input[2]
	local im2a = input[3]
	local im2b = input[4]
        local temp = input[5]

	-- ds
	local dsampled_image = gradOutput[1][1]
	local dbaseline = gradOutput[1][2]
		

	--backprop through baseline
	--do not continue back-backpropagating through players 
	self.baseline:backward(torch.CudaTensor(1,1),dbaseline)

	--backrprop through image selection
        local dprediction = self.image_selection:backward({self.prediction, temp}, dsampled_image)

	--backprop through player 2 
	local dsampled_feat = self.player2:backward({im2a, im2b, self.sampled_feat}, dprediction)
	
	-- backprop though selection
	local dprobs = self.feature_selection:backward({self.probs, temp}, dsampled_feat)
	
	--backrprop through player 1
	dummy = self.player1:backward({im1a, im1b},dprobs)

	--it doesn't really  matter
	self.gradInputs = dummy


	return self.gradInputs
end

function players:evaluate()
	self.player1:evaluate()
        self.player2:evaluate()
	self.baseline:evaluate()
	self.image_selection:evaluate()
        self.feature_selection:evaluate()
end

function players:training()
	self.player1:training()
        self.player2:training()
        self.baseline:training()
	self.image_selection:training()
	self.feature_selection:training()

end

function players:reinforce(reward)
	self.image_selection:reinforce(reward)
	self.feature_selection:reinforce(reward)
end

function players:parameters()
  	local p1,g1 = self.player1:parameters()
        local p2,g2 = self.player2:parameters()
	local p3,g3 = self.baseline:parameters()
	
	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
  
	local grad_params = {}
 	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end
	for k,v in pairs(g3) do table.insert(grad_params, v) end

	return params, grad_params
end



