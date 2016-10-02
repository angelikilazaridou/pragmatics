require 'torch'
require 'rnn'
require 'dpnn'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoaderCommunication'
require 'misc.DataLoaderGrounding'
require 'misc.optim_updates'
require 'players.Players'
require 'csvigo'
local sender = require 'players.Sender_gr'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Sender and Receiver with grounding')
cmd:text()
cmd:text('Options')

-- Data input settings: Communication
cmd:option('-comm_game','','COMMUNICATION: Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-comm_input_h5','COMMUNICATION: ../DATA/game/v3/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-comm_input_json','COMMUNICATION: ../DATA/game/v3/data.json','path to the json file containing additional info and vocab')
cmd:option('-comm_input_h5_images','COMMUNICATION: ..DATA/game/v3/toy_images.h5','path to the h5 of the referit bounding boxes')
cmd:option('-comm_feat_size',-1,'COMMUNICATION: The number of image features')
cmd:option('-comm_game_size',2,'COMMUNICATION: Number of images in the game')
-- Data input settings: Grounding
cmd:option('-gr_task','','GROUNDING: Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-gr_task_size',1,'GROUNDING: Number of inputs given to the player')
cmd:option('-gr_input_h5','GROUNDING: ../DATA/game/v3/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-gr_input_json','GROUNDING: ../DATA/game/v3/data.json','path to the json file containing additional info and vocab')
cmd:option('-gr_input_h5_images','GROUNDING: ..DATA/game/v3/toy_images.h5','path to the h5 of the referit bounding boxes')
cmd:option('-gr_feat_size',-1,'GROUNDING: The number of image features')
-- Aditional info applicable for both setting
cmd:option('-vocab_size',-1,'The number of words in the vocabulary')
cmd:option('-embeddings_file_S','','The txt file containing the word embeddings for Sender. If this option is used, the word embeddings are going to get initialized')
cmd:option('-embeddings_file_R','','The txt file containing the word embeddings for Receiver. If this option is used, the word embeddings are going to get initialized')
cmd:option('-fine_tune',0,'Option to fine-tune embeddings. 0=no, 1=sender only, 2=receiver only, 3=both')
-- Model parameters
cmd:option('-grounding',0, 'The probability of switching to a grounding task [0=just communication-1=just grounding].')
cmd:option('-hidden_size',20,'The hidden size ')
cmd:option('-scale_output',0,'Whether to add a sigmoid at the output of the model')
cmd:option('-dropout',0,'Dropout in the visual input')
cmd:option('-property_size', -1, 'The size of the property latent space')
cmd:option('-embedding_size_R',-1,'The size of the embeddings that the Receiver uses. If set to -1, use the dimensionality of the loaded word vectors.')
cmd:option('-embedding_size_S',-1,'The size of the embeddings that the Sender uses. If set to -1, use the dimensionality of the loaded word vectors.')
-- Optimization: General
cmd:option('-max_iters', 3500, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',32,'what is the batch size of games')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
-- Optimization: for the model
cmd:option('-temperature',10,'Initial temperature')
cmd:option('-decay_temperature',0.99995,'factor to decay temperature')
cmd:option('-temperature2',1,'Initial temperature 2') -- tried with 0.5, didn't do the job
cmd:option('-anneal_temperature',1.000005,'factor to anneal temperature')
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-learning_rate_decay_start', 2000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 1000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay',0,'Weight decay for L2 norm')
cmd:option('-rewardScale',1,'Scaling alpha of the reward')
-- Evaluation/Checkpointing
cmd:option('-val_images_use', 1000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 3500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'grounding/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 1, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
-- misc
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-verbose',false,'How much info to give')
cmd:option('-print_every',100,'Print some statistics')
cmd:text()


------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
opt.id = '_g@'..opt.comm_game..'_h@'..opt.hidden_size..'_d@'..opt.dropout..'_f@'..opt.comm_feat_size..'_w@'..opt.vocab_size..'_a@'..opt.property_size..'_eS@'..opt.embedding_size_S..'_eR@'..opt.embedding_size_R


torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch' 
  	require 'cunn'
  	cutorch.manualSeed(opt.seed)
  	cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end


------------------------------------------------------------------------------
-- Input data
------------------------------------------------------------------------------

if opt.comm_game == 'v1' then
	opt.comm_input_json = '../DATA/game/v1/data.json'
	opt.comm_input_h5 = '../DATA/game/v1/data.h5'
	opt.comm_input_h5_images = '../DATA/game/v1/vectors_transposed.h5'
elseif opt.comm_game == 'v2' then
	opt.comm_input_json = '../DATA/game/v2/data.json'
        opt.comm_input_h5 = '../DATA/game/v2/data.h5'
        opt.comm_input_h5_images = '../DATA/game/v2/vectors_transposed.h5'
elseif opt.comm_game == 'v3' then
	opt.comm_input_json = '../DATA/game/v3/data.json'
        opt.comm_input_h5 = '../DATA/game/v3/data.h5'
        opt.comm_input_h5_images = '../DATA/game/v3/toy_images.h5'
else
	print('No specific game. Data will be given by user')
end


if opt.gr_task == 'v1' then
        opt.gr_input_json = '../DATA/game/v1/data.json'
        opt.gr_input_h5 = '../DATA/game/v1/data.h5'
        opt.gr_input_h5_images = '../DATA/game/v1/vectors_transposed.h5'
elseif opt.gr_task == 'v2' then
        opt.gr_input_json = '../DATA/game/v2/data.json'
        opt.gr_input_h5 = '../DATA/game/v2/data.h5'
        opt.gr_input_h5_images = '../DATA/game/v2/vectors_transposed.h5'
elseif opt.gr_task == 'v3' then
        opt.gr_input_json = '../DATA/game/v3/data.json'
        opt.gr_input_h5 = '../DATA/game/v3/data.h5'
        opt.gr_input_h5_images = '../DATA/game/v3/toy_images.h5'
else
	print('No specific task. Data will be given by user')
end

local loaderCommunication, loaderGrounding, game_size, feat_size, vocab_size

-------------------------------------------------------------------------------
 -- Create the Data Loader instance for the Communication
-------------------------------------------------------------------------------
if opt.grounding ~= 1 then
	loaderCommunication = DataLoaderCommunication{h5_file = opt.comm_input_h5, json_file = opt.comm_input_json,  feat_size = opt.comm_feat_size, gpu = opt.gpuid, vocab_size = opt.vocab_size, h5_images_file = opt.comm_input_h5_images, game_size = opt.comm_game_size, embeddings_file_S = opt.embeddings_file_S, embeddings_file_R = opt.embeddings_file_R, embedding_size_S = opt.embedding_size_S, embedding_size_R = opt.embedding_size_R}
	game_size = loaderCommunication:getGameSize()
	feat_size = loaderCommunication:getFeatSize()
	vocab_size = loaderCommunication:getVocabSize()
end

------------------------------------------------------------------------------
-- Create the Data Loader instance for the Grounding
-------------------------------------------------------------------------------
if opt.grounding ~= 0 then
	loaderGrounding = DataLoaderGrounding{h5_file = opt.gr_input_h5, json_file = opt.gr_input_json,  feat_size = opt.gr_feat_size, gpu = opt.gpuid, vocab_size = opt.vocab_size, h5_images_file = opt.gr_input_h5_images, embeddings_file_S = opt.embeddings_file_S, embeddings_file_R = opt.embeddings_file_R, embedding_size_S = opt.embedding_size_S, embedding_size_R = opt.embedding_size_R}
	game_size = loaderGrounding:getGameSize()
	feat_size = loaderGrounding:getFeatSize()
	vocab_size = loaderGrounding:getVocabSize()
end


-------------------------------------------------------------------------------
-- Override option to opt
-------------------------------------------------------------------------------
opt.vocab_size = vocab_size
opt.game_size = game_size
opt.feat_size = feat_size

------------------------------------------------------------------------------
-- Printing opt
-----------------------------------------------------------------------------
print(opt)

-------------------------------------------------------------------------------
-- Initialize the network
-------------------------------------------------------------------------------
local protos = {}
local params, grad_params, gr_params, gr_grad_params

-- create communication channel
if opt.grounding < 1 then 

	protos.communication = {}
	protos.communication.players = nn.Players(opt)
	protos.communication.criterion = nn.VRClassReward(protos.communication.players,opt.rewardScale)

	-- if there is embedding file for the sender, initialze embeddings
	if opt.embeddings_file_S~="" then
		local nodes = protos.communication.players.sender:listModules()[1]['forwardnodes']
		for _,node in ipairs(nodes) do
			if node.data.annotations.name=='embeddings_S' then
				node.data.module.weight = loaderCommunication:getEmbeddings("sender").matrix:clone()
			end
		end
	end

	--if there is embedding file for the receiver, initialize embeddings
	if opt.embeddings_file_R~="" then
        	local nodes = protos.communication.players.receiver:listModules()[1]['forwardnodes']
	        for _,node in ipairs(nodes) do
	                if node.data.annotations.name=='embeddings_R' then
        	                node.data.module.weight = loaderCommunication:getEmbeddings("receiver").matrix:t():clone()
                	end
	        end
	end
	
	-- flatten and prepare all model parameters to a single vector after weight sharing
        params, grad_params = protos.communication.players:getParameters()
        params:uniform(-0.08, 0.08)
        assert(params:nElement() == grad_params:nElement())

	if opt.gpuid >= 0 then
		protos.communication.criterion:cuda()
	end


end

-- create grounding channel
if opt.grounding >0 then
	
	protos.grounding = {}
	protos.grounding.players = sender.model(opt.gr_task_size, opt.gr_feat_size, opt.vocab_size, opt.property_size, opt.embedding_size_S, opt.dropout, opt.gpuid)
	protos.grounding.criterion = nn.CrossEntropyCriterion()

	-- flatten and prepare all model parameters to a single vector.
        gr_params, gr_grad_params = protos.grounding.players:getParameters()
        gr_params:uniform(-0.08, 0.08)
        assert(gr_params:nElement() == gr_grad_params:nElement())
	
	--ship to gpu
	if opt.gpuid >= 0 then
        	protos.grounding.criterion:cuda()
	end
	
end


collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
	local verbose = utils.getopt(evalopt, 'verbose', true)
	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

	protos.communication.players:evaluate() 
	loaderCommunication:resetIterator(split) -- rewind iteator back to first datapoint in the split
  	
	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local acc = 0
	while true do

		-- get batch of data  
		local data = loaderCommunication:getBatch{batch_size = opt.batch_size, split = 'val'}


		local inputsS = {}
		--insert images
		for i=1,#data.images do
			table.insert(inputsS,data.images[i])
		end
		--insert the shuffled refs for P2 	
		local inputsR = {}
		for i=1,#data.refs do
			table.insert(inputsR, data.refs[i])
	    	end
    
		--forward model
		local outputs = protos.communication.players:forward({inputsS, inputsR, opt.temperature})
   
		--prepage gold data	
		local gold = data.referent_position
    	
		--forward loss
		local loss = protos.communication.criterion:forward(outputs, gold)
		for k=1,opt.batch_size do
			if outputs[1][k][gold[k][1]]==1 then
				acc = acc+1
			end
		end

	 	--average loss
		loss_sum = loss_sum + loss
		loss_evals = loss_evals + 1
    		n = n+opt.batch_size
	
		-- if we wrapped around the split or used up val imgs budget then bail
		local ix0 = data.bounds.it_pos_now
		local ix1 = math.min(data.bounds.it_max, val_images_use)	
		if verbose then
			print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
		end	

		if loss_evals % 10 == 0 then collectgarbage() end	
		if data.bounds.wrapped then break end -- the split ran out of data, lets break out
		if n >= val_images_use then break end -- we've used enough images	
	end

	return loss_sum/loss_evals, acc/(loss_evals*opt.batch_size)

end

--------------------------------------------------------------------------------
-- Supervised objective
--------------------------------------------------------------------------------
local function groundingLoss()

	protos.grounding.players:training()
	gr_grad_params:zero()

	----------------------------------------------------------------------------
        -- Forward pass
        ----------------------------------------------------------------------------
	local d = loaderGrounding:getBatch{batch_size = opt.batch_size, split = 'train'}

	-- forward model
	local outputs = protos.grounding.players:forward(d.images)
	
	-- compute loss
	local loss = protos.grounding.criterion:forward(outputs, d.labels)

	----------------------------------------------------------------------------
	-- Backward pass
	----------------------------------------------------------------------------
	-- backprop through criterion
	local dpredicted = protos.grounding.criterion:backward(outputs, d.labels)

        -- backprop through model
        local dummy = protos.grounding.players:backward(d.images, dpredicted)

	-- TODO: not sure if this is correct
	-- clip gradients
	gr_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	return loss
end
-------------------------------------------------------------------------------
-- RL objective
-------------------------------------------------------------------------------
local iter = 0
local function communicationLoss()

	protos.communication.players:training()
	grad_params:zero()

	----------------------------------------------------------------------------
	-- Forward pass
	-----------------------------------------------------------------------------

	-- get batch of data  
	local data = loaderCommunication:getBatch{batch_size = opt.batch_size, split = 'train'}
  
  
	local inputsS = {}
        --insert images
        for i=1,#data.images do
  	      table.insert(inputsS,data.images[i])
        end
        --insert the shuffled refs for P2       
        local inputsR = {}
        for i=1,#data.refs do
              table.insert(inputsR, data.refs[i])
        end

	--forward model
	local outputs = protos.communication.players:forward({inputsS, inputsR, opt.temperature})
  	
	--compile gold data
	local gold = data.referent_position
	
	--forward in criterion to get loss
	local loss = protos.communication.criterion:forward(outputs, gold)

	--print(torch.sum(gold,2))
	-----------------------------------------------------------------------------
	-- Backward pass
	-----------------------------------------------------------------------------
	-- backprop through criterion
	local dpredicted = protos.communication.criterion:backward(outputs, gold)

	-- backprop through model
	local dummy = protos.communication.players:backward({inputsS, inputsR, opt.temperature}, {dpredicted})
	
	-- freeze during backprop for sender
	if opt.embeddings_file_S~="" and (opt.fine_tune==0 or opt.fine_tune==2) then
        	local nodes = protos.communication.players.sender:listModules()[1]['backwardnodes']
        	for _,node in ipairs(nodes) do
                	if node.data.annotations.name=='embeddings_S' then
                       		node.data.module.gradWeight:fill(0)
                	end
        	end
	end
	-- freeze during backprop for receiver
        if opt.embeddings_file_R~="" and (opt.fine_tune==0 or opt.fine_tune==1) then
                local nodes = protos.communication.players.receiver:listModules()[1]['backwardnodes']
                for _,node in ipairs(nodes) do
                        if node.data.annotations.name=='embeddings_R' then
				node.data.module.gradWeight:fill(0)
                        end
                end
        end


	-- clip gradients
	grad_params:clamp(-opt.grad_clip, opt.grad_clip)


	return loss
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local gr_loss=0
local optim_state = {}
local gr_optim_state = {}
local val_acc_history = {}
local loss_history = {}
local best_score=nil
local checkpoint_path = opt.checkpoint_path .. 'cp_id' .. opt.id ..'.cp'



while true do  

	local losses
	local coin = torch.uniform()
	if coin < opt.grounding then
		if opt.grounding ~= 1 then
			gr_params[{{gr_params:nElement()-(opt.embedding_size_S*opt.vocab_size),gr_params:nElement()}}] = params[{{params:nElement()-(opt.embedding_size_S*opt.vocab_size),params:nElement()}}]:clone()
		end
		gr_loss = groundingLoss()
	else
		if opt.grounding ~= 0 then
			params[{{params:nElement()-(opt.embedding_size_S*opt.vocab_size),params:nElement()}}] = gr_params[{{gr_params:nElement()-(opt.embedding_size_S*opt.vocab_size),gr_params:nElement()}}]:clone()
		end
		losses = communicationLoss()
	end

	local loss = 0
	local acc = 0

	if iter % opt.print_every == 0  and opt.grounding~=1 then
		--evaluate val performance 
		loss,acc = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})
		val_acc_history[iter] = acc
		loss_history[iter] = losses
	end

 	-- save checkpoint once in a while (or on final iteration)
 	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    		-- write a (thin) json report
    		local checkpoint = {}
    		checkpoint.opt = opt
    		checkpoint.iter = iter
    		checkpoint.loss_history = loss_history
    		checkpoint.val_acc_history = val_acc_history

    		utils.write_json(checkpoint_path .. '.json', checkpoint)
    		--print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    		-- write the full model checkpoint as well if we did better than ever
    		local current_score = loss
    		if iter > 0 then -- dont save on very first iteration
			-- include the protos (which have weights) and save to filE
			local save_protos = {}
			save_protos.communication = protos.communication.players -- these are shared clones, and point to correct param storage
			if opt.grounding > 0 then
				save_protos.grounding = protos.grounding.sender
			end
			checkpoint.protos = save_protos
			-- also include the vocabulary mapping so that we can use the checkpoint 
 			-- alone to run on arbitrary images without the data loader
			torch.save(checkpoint_path .. '.t7', checkpoint)
			--print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
		end
	end
	-- decay the learning rate
	local learning_rate = opt.learning_rate
	if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
		local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    		local decay_factor = math.pow(0.5, frac)
		learning_rate = learning_rate * decay_factor -- set the decayed rate
 	 end
	--anneal temperature
	if opt.temperature >0 then
  		opt.temperature = math.max(0.000001,opt.decay_temperature * opt.temperature)
	else
		opt.temperature = 1
	end
	--opt.temperature2 = math.min(1,opt.anneal_temperature * opt.temperature2)

	if iter % opt.print_every == 0 then
		--print(string.format("%d, grad norm = %6.4e, param norm = %6.4e, grad/param norm = %6.4e, lr = %6.4e", iter, grad_params:norm(), params:norm(), grad_params:norm() / params:norm(), learning_rate))
 		print(string.format("%d @ %f @ %f",iter, acc, gr_loss))
 	 end
	

	-- perform a parameter update
	if coin > opt.grounding then
		if opt.optim == 'rmsprop' then
			rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
		elseif opt.optim == 'adagrad' then
 			adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
		elseif opt.optim == 'sgd' then
 			sgd(params, grad_params, opt.learning_rate)
	 	elseif opt.optim == 'sgdm' then
 			sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
		elseif opt.optim == 'sgdmom' then
 			sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
	 	elseif opt.optim == 'adam' then
 			adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
	 	else
 			error('bad option opt.optim')
 		end
	else
		if opt.optim == 'rmsprop' then
                        rmsprop(gr_params, gr_grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, gr_optim_state)
                elseif opt.optim == 'adagrad' then
                        adagrad(gr_params, gr_grad_params, learning_rate, opt.optim_epsilon, gr_optim_state)
                elseif opt.optim == 'sgd' then
                        sgd(gr_params, gr_grad_params, opt.learning_rate)
                elseif opt.optim == 'sgdm' then
                        sgdm(gr_params, gr_grad_params, learning_rate, opt.optim_alpha, gr_optim_state)
                elseif opt.optim == 'sgdmom' then
                        sgdmom(gr_params, gr_grad_params, learning_rate, opt.optim_alpha, gr_optim_state)
                elseif opt.optim == 'adam' then
                        adam(gr_params, gr_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon,gr_optim_state)
                else
                        error('bad option opt.optim')
                end

	end
 	-- stopping criterions
 	iter = iter + 1
 	if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
 	if loss0 == nil then loss0 = losses end

	--if losses > loss0 * 20 then
	--print(string.format('loss seems to be exploding, quitting. %f vs %f', losses, loss0))
  	--  break
 	--end

	if opt.max_iters+1 > 0 and iter >= opt.max_iters+1 then break end -- stopping criterion

end
