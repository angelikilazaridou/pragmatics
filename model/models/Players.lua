require 'dpnn'

local player1 = require 'models.player1'
require 'models.Oracle_P2'

local players, parent =  torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	-- params
	self.batch_size = opt.batch_size
	self.vocab_size = opt.vocab_size
	self.crit = opt.crit
	self.game_size = opt.game_size
	--defining the two players
	self.player1 = player1.model(opt.game_size, opt.feat_size, opt.vocab_size, opt.hidden_size, opt.scale_output, opt.gpuid, opt.k) 
	self.player2 = nn.Oracle_P2(self.batch_size, self.vocab_size)
	-- categorical for selection of feature
	self.selection = nn.ReinforceCategorical(true):cuda()
	-- baseline 
	self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1)):cuda()


end

--called from game:backward()
function players:updateGradInput(input, gradOutput)

	local dprediction, dbaseline, dfeats_SM

	if self.crit == 'reward' then
		dprediction = gradOutput[1][1] -- from VRClassReward, this is zero
                dbaseline = gradOutput[1][2]
		dfeats_SM = torch.CudaTensor(self.batch_size, self.vocab_size):fill(0)
        elseif self.crit == 'MSE' then
		dprediction = torch.CudaTensor(self.batch_size, self.game_size):fill(0)
                dbaseline = torch.CudaTensor(self.batch_size,1):fill(0)
		dfeats_SM = gradOutput[1]
        else
		dfeats_SM = gradOutput[1]
               	dbaseline = gradOutput[2][2]
		dprediction = gradOutput[2][1] --  but it's 0 really
        end

	--backprop through baseline
	--do not continue back-backpropagating through players 
	self.baseline:backward(torch.CudaTensor(1,1),dbaseline)

	--backprop through player 2 is not really possible, just need to generate a 0 gradient for the selection module
	--the doutput_p2 are basically zeros
	local dsampled_feats = torch.CudaTensor(self.batch_size, self.vocab_size):fill(0)
	dfeats_SM = dfeats_SM + self.selection:backward(self.feats_SM, dsampled_feats)

	--backrprop through player 1
	dummy = self.player1:backward({input[1],input[2]},{dfeats_SM, dsampled_feats})
	--it doesn't really  matter
	self.gradInputs = dummy


	return self.gradInputs
end



--called from game:forward()
function players:updateOutput(input)

	-- input:
	-- 1-2 images for P1
	-- 3-4 refs for P2

	--player 1 receives 2 images --
	-- does a forward and gives back 1 action -> 1 feature
	local outputs = self.player1:forward({input[1],input[2]})
	self.feats_logSM = outputs[1]
	self.feats_SM = outputs[2]
	self.sampled_feat = self.selection:forward(self.feats_SM)

	print('Sampled feature')
	print(self.sampled_feat)

	-- player 2 receives 2 images and 1 feature
	-- does a forward pass and gives back the id of the image
	local prediction = self.player2:forward({input[3], input[4], self.sampled_feat})

	--need also the baseline 
	local baseline = self.baseline:forward(torch.CudaTensor(self.batch_size,1))

	if self.crit == 'reward' then
		self.output = {prediction, baseline}
	elseif self.crit == 'MSE' then
		self.output = self.feats_logSM
	else
		self.output = {self.feats_logSM, {prediction, baseline}}
	end
		
	return self.output
end


function players:evaluate()
	for k,v in pairs(self.player1) do v:evaluate() end
	for k,v in pairs(self.baseline) do v:evaluate() end
	for k,v in pairs(seld.selection) do v:evaluate() end
end

function players:training()
	for k,v in pairs(self.player1) do v:training() end
        for k,v in pairs(self.baseline) do v:training() end
        for k,v in pairs(seld.selection) do v:training() end

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



