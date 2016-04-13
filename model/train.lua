
require 'torch'
require 'rnn'
require 'dpnn'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
require 'models.Players'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Model vs Oracle')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-game_session','','Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-input_h5','../DATA/game/v3/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','../DATA/game/v3/data.json','path to the json file containing additional info and vocab')
cmd:option('-input_h5_images','..DATA/game/v3/toy_images.h5','path to the h5 of the referit bounding boxes')
cmd:option('-feat_size',-1,'The number of image features')
cmd:option('-vocab_size',-1,'The number of properties')
cmd:option('-game_size','2','Number of images in the game')
-- Select model
cmd:option('-crit','reward_discr','What criterion to use')
cmd:option('-hidden_size',20,'The hidden size of the discriminative layer')
cmd:option('-scale_output',0,'Whether to add a sigmoid at the output of the model')
-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',32,'what is the batch size of games')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
-- Optimization: for the model
cmd:option('-temperature',10,'Initial temperature')
cmd:option('-decay_temperature',0.99995,'factor to decay temperature')
cmd:option('-temperature2',1,'Initial temperature 2') -- tried with 0.5, didn't do the job
cmd:option('-anneal_temperature',1.000005,'factor to anneal temperature')
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-learning_rate_decay_start', 20000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 200000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay',0,'Weight decay for L2 norm')
cmd:option('-rewardScale','1','Scaling alpha of the reward')
-- Evaluation/Checkpointing
cmd:option('-val_images_use', 100, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 100, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'test/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
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
opt.id = '_h@'..opt.hidden_size..'_k@'..'_scOut@'..opt.scale_output..'_w@'..opt.weight_decay..'_lr@'..opt.learning_rate..'_dlr@'..opt.learning_rate_decay_every


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

if opt.game_session == 'v1' then
	opt.input_json = '../DATA/game/v1/data.json'
	opt.input_h5 = '../DATA/game/v1/data.h5'
	opt.input_h5_images = '..DATA/game/v1/ALL_REFERIT.h5'
elseif opt.game_session == 'v2' then
	opt.input_json = '../DATA/game/v2/data.json'
        opt.input_h5 = '../DATA/game/v2/data.h5'
        opt.input_h5_images = '..DATA/game/v2/images_single.h5'
elseif opt.game_session == 'v3' then
	opt.input_json = '../DATA/game/v3/data.json'
        opt.input_h5 = '../DATA/game/v3/data.h5'
        opt.input_h5_images = '..DATA/game/v3/toy_images.h5'
else
	print('No specific game. Data will be given by user')
end


-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json,  feat_size = opt.feat_size, gpu = opt.gpuid, vocab_size = opt.vocab_size, h5_images_file = opt.input_h5_images, game_size = opt.game_size}
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

print(string.format('Parameters are game_size=%d feat_size=%d, vocab_size=%d\n', game_size, feat_size,vocab_size))
protos.players = nn.Players(opt)

if opt.crit == 'reward_discr' then
  protos.criterion = nn.VRClassReward(protos.players,opt.rewardScale)
else
	print(string.format('Wrog criterion: %s\n',opt.crit))
end

-- ship criterion to GPU, model is shipped dome inside model
if opt.gpuid >= 0 then
	--model is shipped to cpu within the model
	protos.criterion:cuda()
end


-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.players:getParameters()
params:uniform(-0.08, 0.08) 
print('total number of parameters in Game: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.players:evaluate() 
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  	
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local acc = 0
	while true do

  	-- get batch of data  
    local data = loader:getBatch{batch_size = opt.batch_size, split = 'train'}


    local inputs = {}
    --insert images
    for i=1,#data.images do
     	table.insert(inputs,data.images[i])
    end
    --insert the shuffled refs for P2 
    for i=1,#data.refs do
	table.insert(inputs, data.refs[i])
    end
    --insert temperature
    table.insert(inputs,opt.temperature)
    
    --forward model
    local outputs = protos.players:forward(inputs)
   
    local nodes = protos.players.player1:listModules()[1]['forwardnodes']
for _,node in ipairs(nodes) do
 if node.data.annotations.name=='property' then
    extended_dot_vector = node.data.module.weight
    print('tralala')
    print(extended_dot_vector)
 end
end
 
    --prepage gold data
    local gold
    if opt.crit == 'reward_discr' then
      --gold = data.single_discriminative
      gold = data.referent_position
    end
    
    --forward loss
    local loss = protos.criterion:forward(outputs, gold)

    for k=1,opt.batch_size do
      if outputs[1][k][gold[k][1]]==1 then
        acc = acc+1
      end
    end
    --print(torch.sum(gold,2))
    --print(loss)

 
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

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()

	protos.players:training()
  grad_params:zero()

  ----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------

  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train'}
  
  
	local inputs = {}
	--compile input to players
	for i=1,#data.images do
		table.insert(inputs,data.images[i])
	end
	--insert the shuffled refs for P2 
	for i=1,#data.refs do
        	table.insert(inputs, data.refs[i])
    	end
	--insert temperature
	table.insert(inputs,opt.temperature)
	--forward model
	local outputs = protos.players:forward(inputs)
  	
	--compile gold data
	local gold
	if opt.crit == 'reward_discr' then
		--gold = data.single_discriminative
		gold = data.referent_position
	end

	--forward in criterion to get loss
  local loss = protos.criterion:forward(outputs, gold)

	--print(torch.sum(gold,2))
	-----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop through criterion
  local dpredicted = protos.criterion:backward(outputs, gold)

  -- backprop through model
  local dummy = protos.players:backward(inputs, {dpredicted})

  -- clip gradients
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)


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

 	-- save checkpoint once in a while (or on final iteration)
 	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

  	print(string.format('Training loss %f',losses))
    -- evaluate the validation performance
    local loss,acc = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})
    print(string.format('VALIDATION loss: %f and prediction accuracy %f and temperature %f temperature2 %f', loss, acc, opt.temperature, opt.temperature2))

		--keep test score for now
    val_acc_history[iter] = loss
    val_prop_acc_history[iter] = loss

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
    local current_score = loss
    
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

  --anneal temperature
  opt.temperature = math.max(0.000001,opt.decay_temperature * opt.temperature)
  opt.temperature2 = math.min(1,opt.anneal_temperature * opt.temperature2)

	if iter % opt.print_every == 0 then
    print(string.format("%d, grad norm = %6.4e, param norm = %6.4e, grad/param norm = %6.4e, lr = %6.4e", iter, grad_params:norm(), params:norm(), grad_params:norm() / params:norm(), learning_rate))
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
  --if losses > loss0 * 20 then
  --	print(string.format('loss seems to be exploding, quitting. %f vs %f', losses, loss0))
  --  break
  --end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
