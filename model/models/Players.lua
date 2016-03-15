require 'dp'

local player1 = require 'models.player1'
local player2 = require 'models.oracle_p2'

local players, parent torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	--defining the two players
	self.player1 = player1.model(opt.game_size, opt.feat_size, opt.vocab_size, opt.hidden_size, opt.scale_output, opt.gpuid, opt.k) 
	self.player2 = player2()
	self.baseline = nn.Linear(opt.feat_size,1)

end

--called from game:backward()
function players:updateGradInput(input, gradOutput)

	dprediction = gradOutput[1] -- from VRClassReward, this is zero
	dbaseline = gradOutput[2]


	--backprop through baseline
	--do not continue back-backpropagating through players 
	self.baseline:backward(self.probs,dbaseline)

	--backprop through player 2
	doutput_p2 = self.player2:backward({input, self.sampled_feats}, dprediction)
	dsampled_feats = doutput_p2[2]

	--backrprop through player 1
	dummy = self.player1:backward(input,dsampled_feats)
	--it doesn't really  matter
	self.gradInputs = dummy


	return self.gradInputs
end



--called from game:forward()
function players:updateOutput(input)

	--player 1 receives 2 images
	-- does a forward and gives back 1 action -> 1 feature
	local outputs = self.player1:forward(input)
	self.probs = outputs[1]
	self.sampled_feats = nn.ReinforceCategorical(false)(self.probs)


	--player 2 receives 2 images and 1 feature
	-- does a forward pass and gives back the id of the image
	local prediction = self.player2:forward({input, self.sampled_feats})

	--need also the baseline reward
	local baseline = self.baseline:forward(self.probs)

	self.output = {prediction, baseline}
		
	return self.output
end


function players:evaluate()
	for k,v in pairs(self.player1) do v:evaluate() end
end

function players:training()
	for k,v in pairs(self.player1) do v:training() end
end

function players:parameters()
  	local p1,g1 = self.player1:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
  
	local grad_params = {}
 	for k,v in pairs(g1) do table.insert(grad_params, v) end

	return params, grad_params
end



