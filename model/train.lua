require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
local XOR1 = require 'models.XOR1'
local XOR2 = require 'models.XOR2'
local XOR3 = require 'models.XOR3'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a XOR. Why? Because :)')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','../DATA/XOR/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','../DATA/XOR/data.json','path to the json file containing additional info and vocab')
-- Select model
cmd:option('-model','XOR2','What model to use')
cmd:option('-crit','MSE','What criterion to use')
-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
-- Optimization: for the model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 500, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
	require 'cutorch'
  	require 'cunn'
  	cutorch.manualSeed(opt.seed)
  	cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, label_format = opt.crit}
local game_size = loader:getGameSize()
local feat_size = loader:getFeatSize()
local vocab_size = loader:getVocabSize()


-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
local to_share = 0

-- create protos from scratch
if opt.model == 'XOR1' then
	print(string.format('Parameters are game_size=%d feat_size=%d, to_share=%d\n',game_size, feat_size,to_share))
	protos.xor = XOR1.xor(game_size, feat_size, to_share)
elseif opt.model == 'XOR2' then
	print(string.format('Parameters are game_size=%d feat_size=%d, to_share=%d\n',game_size, feat_size,to_share))
        protos.xor = XOR2.xor(game_size, feat_size)
else
	print(string.format('Parameters are game_size=%d feat_size=%d, to_share=%d\n',game_size, feat_size,to_share))
        protos.xor = XOR3.xor(game_size, feat_size)
end

--add criterion
if opt.crit == 'NLL' then
	protos.criterion = nn.ClassNLLCriterion()
elseif opt.crit == 'MSE' then
	protos.criterion = nn.MSECriterion()
else
	print('Wrong criterion')
end


-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  	for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.xor:getParameters()
print('total number of parameters in LM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())


collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  	local verbose = utils.getopt(evalopt, 'verbose', true)
  	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  	protos.xor:evaluate()
  	loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  	local n = 0
  	local loss_sum = 0
  	local loss_evals = 0
  	local predictions = {}
  	local correct = 0
	local all = 0
	while true do

    		-- fetch a batch of data
    		local data = loader:getBatch{batch_size = opt.batch_size, split = split}

    		-- forward the model to get loss
    		local logprobs = protos.xor:forward({unpack(data.images)})
		
    		local loss = protos.criterion:forward(logprobs, data.labels)
		
    		loss_sum = loss_sum + loss
    		loss_evals = loss_evals + 1

		local _,predicted = torch.max(logprobs,2)
        	local gold
        	if opt.crit == 'MSE' then
                	_,gold = torch.max(data.labels,2)
        	elseif opt.crit == 'NLL' then
                	-- add a dummy dimension 
                	gold = data.labels:view(1,data.labels:size(1)):t()
        	else
                	print(string.format('Wrong criterion: %s',opt.crit))
        	end
        	for i=1,predicted:size(1) do
                	if predicted[{i,1}] == gold[{i,1}] then
                        	 correct = correct +1
                	end
                	all = all+1
        	end
	
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

	return correct/all
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
	protos.xor:training()
  	grad_params:zero()

	local all = 0
	local correct =0
  	-----------------------------------------------------------------------------
  	-- Forward pass
  	-----------------------------------------------------------------------------
  	-- get batch of data  
  	local data = loader:getBatch{batch_size = opt.batch_size, split = 'train'}
  
  	-- forward the model on images (most work happens here)
	local inputs = {}
	for i=1,#data.images do
		table.insert(inputs,data.images[i])
	end
	local logprobs = protos.xor:forward(inputs)
  	-- forward the language model criterion
  	local loss = protos.criterion:forward(logprobs, data.labels)


	--compute accuracy
	local _,predicted = torch.max(logprobs,2)
	local gold
	if opt.crit == 'MSE' then
		_,gold = torch.max(data.labels,2)
	elseif opt.crit == 'NLL' then
		-- add a dummy dimension 
		gold = data.labels:view(1,data.labels:size(1)):t()
	else
		print(string.format('Wrong criterion: %s',opt.crit))
        end
	for i=1,predicted:size(1) do
		if predicted[{i,1}] == gold[{i,1}] then
                       	 correct = correct +1
                end
                all = all+1
        end
  
  	-----------------------------------------------------------------------------
  	-- Backward pass
  	-----------------------------------------------------------------------------
  	-- backprop criterion
  	local dlogprobs = protos.criterion:backward(logprobs, data.labels)
  	-- backprop language model
	table.insert(inputs,data.labels)
  	local dummy = unpack(protos.xor:backward(inputs, dlogprobs))

  	-- clip gradients
  	-- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  	grad_params:clamp(-opt.grad_clip, opt.grad_clip)


  	-- and lets get out!
  	return loss,correct/all
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
local checkpoint_path = opt.checkpoint_path .. 'cp_id' .. opt.id ..'.cp'

while true do  

  	-- eval loss/gradient
  	local losses, acc = lossFun()
  	if iter % opt.losses_log_every == 0 then loss_history[iter] = losses end
  	print(string.format('iter %d: %f %f', iter, losses,acc))

  	-- save checkpoint once in a while (or on final iteration)
  	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    		-- evaluate the validation performance
    		local val_loss = eval_split('val', {val_images_use = opt.val_images_use})
    		print('validation loss: ', val_loss)
    		val_loss_history[iter] = val_loss

    		-- write a (thin) json report
    		local checkpoint = {}
    		checkpoint.opt = opt
    		checkpoint.iter = iter
    		checkpoint.loss_history = loss_history
    		checkpoint.val_loss_history = val_loss_history
    		checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval

    		utils.write_json(checkpoint_path .. '.json', checkpoint)
    		print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    		-- write the full model checkpoint as well if we did better than ever
    		local current_score
    
		if best_score == nil or current_score > best_score then
      			best_score = current_score
      			if iter > 0 then -- dont save on very first iteration
        			-- include the protos (which have weights) and save to file
        			local save_protos = {}
        			save_protos.xor = protos.xor -- these are shared clones, and point to correct param storage
        			checkpoint.protos = save_protos
        			-- also include the vocabulary mapping so that we can use the checkpoint 
        			-- alone to run on arbitrary images without the data loader
        			torch.save(checkpoint_path .. '.t7', checkpoint)
        			print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      			end
    		end
	end

  	-- decay the learning rate for both LM and CNN
  	local learning_rate = opt.learning_rate
  	if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    		local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    		local decay_factor = math.pow(0.5, frac)
    		learning_rate = learning_rate * decay_factor -- set the decayed rate
  	end

  	-- perform a parameter update
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


  	-- stopping criterions
  	iter = iter + 1
  	if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  	if loss0 == nil then loss0 = losses end
  	if losses > loss0 * 20 then
    		print('loss seems to be exploding, quitting.')
    		break
  	end
  	if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
