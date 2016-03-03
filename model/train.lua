require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderSingle'
require 'misc.optim_updates'
local model1 = require 'models.model1'
local model1_fast = require 'models.model1_fast'
local model1_fast_NB = require 'models.model1_fast_NB'
local model1_fast_NB_hl = require 'models.model1_fast_NB_hl'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a XOR. Why? Because :)')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','../DATA/visAttCarina/processed/0shot_single_test/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','..//DATA/visAttCarina/processed/0shot_single_test/data.json','path to the json file containing additional info and vocab')
cmd:option('-feat_size',-1,'The number of image features')
cmd:option('-vocab_size',-1,'The number of properties')
cmd:option('-single_images',1,'Whether to train on centroid or not')
-- Select model
cmd:option('-model','model1_fast','What model to use')
cmd:option('-crit','MSE','What criterion to use')
cmd:option('-hidden_size',20,'The hidden size of the discriminative layer')
cmd:option('-k',1,'The slope of sigmoid')
cmd:option('-scale_output',0,'Whether to add a sigmoid at teh output of the model')
-- Optimization: General
cmd:option('-max_iters', 7000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
-- Optimization: for the model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.001,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 500, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay',0,'Weight decay for L2 norm')
-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 100, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'tune/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-beta',1,'beta for f_x')
-- misc
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-verbose',false,'How much info to give')
cmd:option('-print_every',1000,'Print some statistics')
cmd:text()


------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
opt.id = '_'..opt.model..'_h@'..opt.hidden_size..'_k@'..'_scOut@'..opt.scale_output..'_w@'..opt.weight_decay..'_lr@'..opt.learning_rate..'_dlr@'..opt.learning_rate_decay_every


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
local loader 
if opt.single_images==0 then
	loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, label_format = opt.crit, feat_size = opt.feat_size, gpu = opt.gpuid, vocab_size = opt.vocab_size}
else
	loader =  DataLoaderSingle{h5_file = opt.input_h5, json_file = opt.input_json, label_format = opt.crit, feat_size = opt.feat_size, gpu = opt.gpuid, vocab_size = opt.vocab_size}
end
local game_size = loader:getGameSize()
local feat_size = loader:getFeatSize()
local vocab_size = loader:getVocabSize()


-------------------------------------------------------------------------------
-- Override option to opt
-------------------------------------------------------------------------------
opt.vocab_size = vocab_size
opt.game_size = game_size
opt.feat_size = feat_size


-------------------------------------------------------------------------------
-- Initialize the network
-------------------------------------------------------------------------------
local protos = {}

print(string.format('Parameters are model=%s game_size=%d feat_size=%d, vocab_size=%d\n',opt.model, game_size, feat_size,vocab_size))
-- create protos from scratch
if opt.model == 'model1_fast' then
        protos.model = model1_fast.model(game_size, feat_size, vocab_size, opt.hidden_size, opt.scale_output, opt.gpuid, opt.k)
elseif opt.model == 'model1_fast_NB' then
        protos.model = model1_fast_NB.model(game_size, feat_size, vocab_size, opt.hidden_size, opt.scale_output, opt.gpuid, opt.k)
elseif opt.model == 'model1_fast_NB_hl' then
        protos.model = model1_fast_NB_hl.model(game_size, feat_size, vocab_size, opt.hidden_size, opt.scale_output, opt.gpuid, opt.k)
elseif opt.model == 'model1' then
        protos.model = model1.model(game_size, feat_size, vocab_size, opt.hidden_size, to_share, opt.gpuid, opt.k)
else
	print(string.format('Wrong model:%s',opt.model))
end

--add criterion
if opt.crit == 'MSE' then
	protos.criterion = nn.MSECriterion()
else	
	print(string.format('Wrong criterion: %s',opt.crit))
end

-- ship criterion to GPU, model is shipped dome inside model
if opt.gpuid >= 0 then
	--model is shipped to cpu within the model
	protos.criterion:cuda()
end

-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.model:getParameters()
params:uniform(-0.08, 0.08) 
print('total number of parameters in LM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

--parameters to regularize
reg = {}
reg[1] = protos.model.forwardnodes[5].data.module.weight --Linear mapping
reg[2] = protos.model.forwardnodes[5].data.module.bias
reg[3] = protos.model.forwardnodes[13].data.module.weight --hidden for XOR
reg[4] = protos.model.forwardnodes[13].data.module.bias
reg[5] = protos.model.forwardnodes[16].data.module.weight -- decision
reg[6] = protos.model.forwardnodes[16].data.module.bias


collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  	local verbose = utils.getopt(evalopt, 'verbose', true)
  	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  	protos.model:evaluate()
  	loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  	local n = 0
  	local loss_sum = 0
  	local loss_evals = 0
  	local predictions = {}
	 --for discriminativeness
        local TP_D = 0
        local TP_FN_D = 0
        local TP_FP_D = 0

	local all = 0
	local balance = 0
	-- for features
	local TP = torch.zeros(1,2)
	local TP_FN = torch.zeros(1,2)
	local TP_FP = torch.zeros(1,2)

	local sparsity = 0	
	while true do

    		-- fetch a batch of data
    		local data = loader:getBatch{batch_size = opt.batch_size, split = split}

    		-- forward the model to get loss
    		local outputs = protos.model:forward({unpack(data.images)})
		local predicted = outputs[1]

    		local loss = protos.criterion:forward(predicted, data.labels)
		
		--average ;oss
    		loss_sum = loss_sum + loss
    		loss_evals = loss_evals + 1

		--compute accuracy
		local predicted2 = predicted:clone()
		predicted2:apply(function(x) if x>0.5 then return 1 else return 0 end end)
		
		--get in a tensor the properties
		local properties = {unpack(outputs,2)}
		--print(properties)

		--print(data.properties)
        	for i=1,opt.batch_size do
			--check if prediction of discriminativeness if correct
                        TP_D = TP_D + torch.sum(torch.cmul(predicted2[i],data.labels[i]))
                        TP_FN_D = TP_FN_D + torch.sum(data.labels[i])
                        TP_FP_D = TP_FP_D + torch.sum(predicted2[i])
			sparsity = sparsity + torch.sum(predicted2[i])

			--check if properties are correct
			for ii=1,game_size do
				local s1 = properties[ii][{{i,i},{1,vocab_size}}]:clone():apply(function(x) if x >0 then return 1 else return 0 end end)
				local s11 = properties[ii][{{i,i},{1,vocab_size}}]:clone():apply(function(x) if x >0 then return 0 else return 1 end end)			
				local s2 = data.properties[{{i,i},{ii,ii},{1,vocab_size}}][1]  --3d
				--feature wise
				--compute both 0/1 cases and give the one with bigger F1
				TP[1][1] = TP[1][1] + torch.sum(torch.cmul(s1,s2)) --correct
				TP_FN[1][1] = TP_FN[1][1] + torch.sum(s2) --all relevant
				TP_FP[1][1] = TP_FP[1][1] + torch.sum(s1) -- all predictions
				
				TP[1][2] = TP[1][2] + torch.sum(torch.cmul(s11,s2)) --correct
                                TP_FN[1][2] = TP_FN[1][2] + torch.sum(s2) --all relevant
                                TP_FP[1][2] = TP_FP[1][2] + torch.sum(s11) -- all predictions
					
			
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

	local precision_D = TP_D/TP_FP_D
	local recall_D = TP_D/TP_FN_D
  	local F1_D = ((1+opt.beta^2) * precision_D * recall_D)/((opt.beta^2*precision_D)+recall_D)

	local precision = torch.cdiv(TP,TP_FP)
	local recall = torch.cdiv(TP,TP_FN)
	local F1s = torch.cdiv(torch.mul(torch.cmul(precision,recall),2),torch.add(precision,recall))
	local val,pos = torch.max(F1s,2)
	pos = pos[1][1]

	return precision_D, recall_D, F1_D, precision[1][pos], recall[1][pos], F1s[1][pos], sparsity/all	

end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
	protos.model:training()
  	grad_params:zero()

  	-----------------------------------------------------------------------------
  	-- Forward pass
  	-----------------------------------------------------------------------------
  	-- get batch of data  
  	local data = loader:getBatch{batch_size = opt.batch_size, split = 'train'}
  
  	-- forward the model on images (most work happens here)
	local inputs = {}
	local dinputs = {}

	for i=1,#data.images do
		if opt.gpuid >= 0 then  --gradients for things that we do not care, i.e., the property vectors
			dinputs[i+1] = torch.CudaTensor(opt.batch_size,vocab_size):fill(0)   
		else
			dinputs[i+1] = torch.FloatTensor(opt.batch_size, vocab_size):fill(0)
		end
		table.insert(inputs,data.images[i])
	end
	--getting property vectoes and loss
	local outputs = protos.model:forward(inputs)
	local predicted = outputs[1]
  	-- forward the language model criterion
  	local loss = protos.criterion:forward(predicted, data.labels)

	-----------------------------------------------------------------------------
  	-- Backward pass
  	-----------------------------------------------------------------------------
  	-- backprop criterion
  	local dpredicted = protos.criterion:backward(predicted, data.labels)
  	-- backprop language model
	table.insert(inputs,data.labels)
	dinputs[1] = dpredicted
  	local dummy = unpack(protos.model:backward(inputs, dinputs))

  	-- clip gradients
  	-- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  	grad_params:clamp(-opt.grad_clip, opt.grad_clip)


  	-- and lets get out!
  	return loss
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_acc_history = {}
local val_prop_acc_history = {}
local best_score
local checkpoint_path = opt.checkpoint_path .. 'cp_id' .. opt.id ..'.cp'

while true do  

  	-- eval loss/gradient
  	local losses = lossFun()
  	if iter % opt.losses_log_every == 0 then loss_history[iter] = losses end
  	--print(string.format('iter %d: %f %f', iter, losses,acc))

  	-- save checkpoint once in a while (or on final iteration)
  	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    		-- evaluate the validation performance
    		local precision_D, recall_D, f1_D, precision, recall, f1, sparsity = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})
                print(string.format('VALIDATION (discriminativeness) precision: %f   recall: %f   F1: %f (sparsity: %f)', precision_D, recall_D, f1_D, sparsity))
    		print(string.format('VALIDATION (property) precision: %f   recall: %f   F1: %f', precision, recall, f1))
		
		precision_D, recall_D, f1_D, precision, recall, f1, sparsity = eval_split('test', {val_images_use = opt.val_images_use, verbose=opt.verbose})
                print(string.format('TEST (discriminativeness) precision: %f   recall: %f   F1: %f (sparsity: %f)', precision_D, recall_D, f1_D, sparsity))
		print(string.format('TEST (property) precision: %f   recall: %f   F1: %f', precision, recall, f1))
		
		--check if F1s are nan
		if f1_D~=f1_D then
			f1_D = 0
		end
		--keep test score for now
		val_acc_history[iter] = f1_D
		if f1~=f1 then
			f1 = 0
		end
		val_prop_acc_history[iter] = f1



    		-- write a (thin) json report
    		local checkpoint = {}
    		checkpoint.opt = opt
    		checkpoint.iter = iter
    		checkpoint.loss_history = loss_history
    		checkpoint.val_acc_history = val_acc_history
		checkpoint.val_prop_acc_history = val_prop_acc_history
    		checkpoint.val_predictions = val_predictions 

    		utils.write_json(checkpoint_path .. '.json', checkpoint)
    		print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    		-- write the full model checkpoint as well if we did better than ever
    		local current_score = f1_D
    
		if best_score == nil or current_score >= best_score then
      			best_score = current_score
      			if iter > 0 then -- dont save on very first iteration
        			-- include the protos (which have weights) and save to file
        			local save_protos = {}
        			save_protos.model = protos.model -- these are shared clones, and point to correct param storage
        			checkpoint.protos = save_protos
        			-- also include the vocabulary mapping so that we can use the checkpoint 
        			-- alone to run on arbitrary images without the data loader
        			torch.save(checkpoint_path .. '.t7', checkpoint)
        			print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      			end
    		end
	end

  	-- decay the learning rate
  	local learning_rate = opt.learning_rate
  	if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    		local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    		local decay_factor = math.pow(0.5, frac)
    		learning_rate = learning_rate * decay_factor -- set the decayed rate
  	end

	if iter % opt.print_every == 0 then
        	print(string.format("%d, grad norm = %6.4e, param norm = %6.4e, grad/param norm = %6.4e", iter, grad_params:norm(), params:norm(), grad_params:norm() / params:norm()))
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

	--apply normalization after param update
	if opt.weight_decay >0 then
		for _,w in ipairs(reg) do
      			w:add(-(opt.weight_decay*learning_rate), w)
   		end
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
