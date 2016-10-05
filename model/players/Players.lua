require 'dpnn'
require 'nn'

local sender = require 'players.Sender_old'
local receiver = require 'players.Receiver_old'

local players, parent =  torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	-- params
	self.batch_size = opt.batch_size
	self.vocab_size = opt.vocab_size
	self.game_size = opt.comm_game_size
	self.embedding_size = opt.embedding_size
	self.gpuid = opt.gpuid 

	--defining the two players
	if opt.comm_sender == 'sender_no_embeddings' then
		local s = require 'players.Sender_old_noemb'
		self.sender = s.model(opt.comm_game_size, opt.comm_feat_size, opt.vocab_size, opt.hidden_size, opt.dropout, opt.gpuid)
	elseif opt.comm_sender == 'sender_simple' then
		local s = require 'players.Sender_old'
                self.sender = s.model(opt.comm_game_size, opt.comm_feat_size, opt.vocab_size, opt.embedding_size_S, opt.hidden_size, opt.dropout, opt.gpuid)
	else
		local s = require 'players.Sender_conv'
		self.sender = s.model(opt.comm_game_size, opt.comm_feat_size, opt.vocab_size, opt.embedding_size_S, opt.hidden_size, opt.dropout, opt.gpuid)
	end
	
	self.receiver = receiver.model(opt.comm_game_size, opt.comm_feat_size, opt.vocab_size, opt.property_size, opt.embedding_size_R, opt.hidden_size, opt.dropout, opt.gpuid)

	if self.gpuid == 0 then
		-- categorical for selection of feature
		self.feature_selection = nn.ReinforceCategorical(true):cuda()
		self.image_selection = nn.ReinforceCategorical(true):cuda()
		-- baseline 
	        self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1)):cuda()
		-- for gibbs annealing
		self.annealT = nn.MulConstant(1):cuda()
		self.softmax = nn.SoftMax():cuda()
	else
		self.feature_selection = nn.ReinforceCategorical(true)
		self.image_selection = nn.ReinforceCategorical(true)
		self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1))
		self.annealT = nn.MulConstant(1)
		self.softmax = nn.SoftMax()
	end


end

--called from game:forward()
function players:updateOutput(input)

	-- input: 	
	local inputS = input[1]
	local inputR = input[2]
	local temp = input[3]

	--player 1 receives 2 images --

	-- does a forward and gives back 1 action -> 1 feature
	self.scores = self.sender:forward(inputS)
	-- anneal distribution
	self.annealT.constant_scalar = 1/temp
	self.annealed = self.annealT:forward(self.scores)
	self.probs = self.softmax:forward(self.annealed)
	--sample a feature
	self.sampled_feat = self.feature_selection:forward(self.probs)
	
	to_print = torch.random(1000)
	if to_print%100 == 0 then
		print(torch.sum(self.sampled_feat,1))
	end
	
	-- player 2 receives 2 refs and 1 feature and predicts L or R
	table.insert(inputR, self.sampled_feat)
	self.prediction = self.receiver:forward(inputR)
	-- sample image
	self.sampled_image = self.image_selection:forward(self.prediction)
	-- baseline
	local baseline 
	if self.gpuid == 0 then
		baseline = self.baseline:forward(torch.CudaTensor(self.batch_size,1))
	else	
		baseline = self.baseline:forward(torch.FloatTensor(self.batch_size,1))
	end

	local outputs = {}
	table.insert(outputs, self.sampled_image)
	table.insert(outputs, baseline)
	table.insert(outputs, self.sampled_feat)

	return outputs
end


--called from game:backward()
function players:updateGradInput(input, gradOutput)

	-- input:       
        local inputS = input[1]
        local inputR = input[2]
        local temp = input[3]

	-- ds
	local dsampled_image = gradOutput[1][1]
	local dbaseline = gradOutput[1][2]
		

	--backprop through baseline
	--do not continue back-backpropagating through players 
	if self.gpuid == 0 then
		self.baseline:backward(torch.CudaTensor(1,1),dbaseline)
	else
		self.baseline:backward(torch.FloatTensor(1,1),dbaseline)
	end

	--backrprop through image selection
        local dprediction = self.image_selection:backward(self.prediction, dsampled_image)

	--backprop through player 2 
	table.insert(inputR,self.sampled_feat)
	local dsampled_feat = self.receiver:backward(inputR, dprediction)
	
	-- backprop though selection
	local dprobs = self.feature_selection:backward(self.probs, dsampled_feat)
	
	-- backrpop through softmax annealing
	local dannealed = self.softmax:backward(self.annealed, dprobs)
	local dscores = self.annealT:backward(self.scores, dannealed)
	
	--backrprop through player 1
	dummy = self.sender:backward(inputS,dscores)

	--it doesn't really  matter
	self.gradInputs = dummy


	return self.gradInputs
end

function players:evaluate()
	self.sender:evaluate()
        self.receiver:evaluate()
	self.baseline:evaluate()
	self.image_selection:evaluate()
        self.feature_selection:evaluate()
end

function players:training()
	self.sender:training()
        self.receiver:training()
        self.baseline:training()
	self.image_selection:training()
	self.feature_selection:training()

end

function players:reinforce(reward)
	self.image_selection:reinforce(reward)
	self.feature_selection:reinforce(reward)
end

function players:parameters()
  	local p1,g1 = self.sender:parameters()
	local p2,g2 = self.receiver:parameters()
	local p3,g3 = self.baseline:parameters()

	local params = {}
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
	for k,v in pairs(p1) do table.insert(params, v) end
  
	local grad_params = {}
 	for k,v in pairs(g2) do table.insert(grad_params, v) end
	for k,v in pairs(g3) do table.insert(grad_params, v) end
	for k,v in pairs(g1) do table.insert(grad_params, v) end

	
	return params, grad_params
end



