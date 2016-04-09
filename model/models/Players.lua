require 'dpnn'
require 'nn'

local player1 = require 'models.player1'
require 'models.Oracle_P2'

local players, parent =  torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	-- params
	self.batch_size = opt.batch_size
	self.vocab_size = opt.vocab_size
	self.game_size = opt.game_size

	--defining the two players
	self.player1 = player1.model(opt.game_size, opt.feat_size, opt.vocab_size, opt.hidden_size, opt.gpuid) 
	self.player2 = nn.Oracle_P2(self.batch_size, self.vocab_size)	
	
	if opt.gpuid == 0 then
		-- categorical for selection of feature
		self.selection = nn.ReinforceCategorical(true):cuda()
		-- baseline 
	        self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1)):cuda()
	else
		self.selection = nn.ReinforceCategorical(true)
		self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1))
	end


end

--called from game:forward()
function players:updateOutput(input)

	-- input: 	
	local im1a = input[1]
	local im1b = input[2]
	local ref2a = input[3]
	local ref2b = input[4]
	local temp = input[5]
	
	--player 1 receives 2 images --
	-- does a forward and gives back 1 action -> 1 feature
	self.probs = self.player1:forward({im1a, im1b})
	--sample a feature
	self.sampled_feat = self.selection:forward({self.probs, temp})
	
	-- player 2 receives 2 refs and 1 feature and predicts L or R
	self.prediction = self.player2:forward({ref2a, ref2b, self.sampled_feat})
	--self.prediction = self.sampled_feat
	-- baseline
	local baseline = self.baseline:forward(torch.CudaTensor(self.batch_size,1))

	local outputs = {}
	table.insert(outputs, self.prediction)
	table.insert(outputs, baseline)

	return outputs
end


--called from game:backward()
function players:updateGradInput(input, gradOutput)

	-- input:       
        local im1a = input[1]
        local im1b = input[2]
	local ref2a = input[3]
	local ref2b = input[4]
        local temp = input[5]

	-- ds
	local dprediction = gradOutput[1][1]
	local dbaseline = gradOutput[1][2]

	--backprop through baseline
	--do not continue back-backpropagating through players 
	self.baseline:backward(torch.CudaTensor(1,1),dbaseline)

	--backprop through player 2 is not really possible, just need to generate a 0 gradient for the selection module
	--the doutput_p2 are basically zeros
	local dsampled_feat = torch.CudaTensor(self.batch_size, self.vocab_size):fill(0)

	-- backprop though selection
	local dprobs = self.selection:backward({self.probs, temp}, dsampled_feat)	
	--backrprop through player 1
	dummy = self.player1:backward({im1a, im1b},dprobs)

	--it doesn't really  matter
	self.gradInputs = dummy


	return self.gradInputs
end

function players:evaluate()
	self.player1:evaluate()
	self.baseline:evaluate()
	self.selection:evaluate()
end

function players:training()
	self.player1:training()
    self.baseline:training()
	self.selection:training()

end

function players:reinforce(reward)
	self.selection:reinforce(reward)
end

function players:parameters()
  	local p1,g1 = self.player1:parameters()
	local p2,g2 = self.baseline:parameters()
	
	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
  
	local grad_params = {}
 	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end

	return params, grad_params
end



