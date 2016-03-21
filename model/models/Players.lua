require 'dp'

local player1 = require 'models.player1'
local player2 = require 'models.oracle_p2'

local players, parent torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	--defining the two players
	self.player1 = player1.model(opt.game_size, opt.feat_size, opt.vocab_size, opt.hidden_size, opt.scale_output, opt.gpuid, opt.k) 
	self.player2 = player2()
	self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1))


end

--called from game:backward()
function players:updateGradInput(input, gradOutput)

	local dprediction = gradOutput[1] -- from VRClassReward, this is zero
	local dbaseline 

	if self.crit == 'reward' then
                dbaseline = gradOutput[2]
        elseif self.crit == 'MSE' then
                dbaseline = CudaTensor(1,1):fill(0)
        else
               	dbaseline = gradOutput[2][2]
		dprediction = dprediction+gradOutput[2][1] --  but it's 0 really
        end



	--backprop through baseline
	--do not continue back-backpropagating through players 
	self.baseline:backward(CudaTensor(1,1),dbaseline)

	--backprop through player 2
	doutput_p2 = self.player2:backward({input[3],input[4], self.sampled_feats}, dprediction)
	dsampled_feats = doutput_p2[3]

	--backrprop through player 1
	dummy = self.player1:backward({input[1],input[2]},dsampled_feats)
	--it doesn't really  matter
	self.gradInputs = dummy


	return self.gradInputs
end



--called from game:forward()
function players:updateOutput(input)

	-- input has 2 x 2 images
	-- first part are 1: referent 2:context
	-- second part are shuffled, e.g., 3:context 4: referent 

	--player 1 receives 2 images --
	-- does a forward and gives back 1 action -> 1 feature
	local outputs = self.player1:forward({input[1],input[2]})
	self.probs = outputs[1]
	self.sampled_feats = nn.ReinforceCategorical(false)(self.probs)


	-- player 2 receives 2 images and 1 feature
	-- does a forward pass and gives back the id of the image
	local prediction = self.player2:forward({input[3],input[4], self.sampled_feats})

	--need also the baseline 
	local baseline = self.baseline:forward(CudaTensor(1,1))

	if self.crit == 'reward' then
		self.output = {prediction, baseline}
	elseif self.crit == 'MSE' then
		self.output = {prediction}
	else
		self.output = {prediction, {prediction, baseline}}
	end
		
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
	local p2,g2 = self.baseline:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
  
	local grad_params = {}
 	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end

	return params, grad_params
end



