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
cmd:text('Model vs Model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-game_session','','Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-input_h5','../DATA/game/v3/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','../DATA/game/v3/data.json','path to the json file containing additional info and vocab')
cmd:option('-input_h5_images','..DATA/game/v3/toy_images.h5','path to the h5 of the referit bounding boxes')
cmd:option('-feat_size',4096,'The number of image features')
cmd:option('-vocab_size',10,'The number of properties')
cmd:option('-game_size',2,'Number of images in the game')
-- Select model
cmd:option('-crit','reward_discr','What criterion to use')
cmd:option('-hidden_size',20,'The hidden size of the discriminative layer')
cmd:option('-scale_output',0,'Whether to add a sigmoid at the output of the model')
cmd:option('-dropout',0,'Dropout in the visual input')
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
cmd:option('-learning_rate_decay_start', 20000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 200000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay',0,'Weight decay for L2 norm')
cmd:option('-rewardScale',1,'Scaling alpha of the reward')
-- Evaluation/Checkpointing
cmd:option('-val_images_use', 100, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 3500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'conll/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 1, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
-- misc
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-verbose',false,'How much info to give')
cmd:option('-print_every',1,'Print some statistics')
cmd:text()


------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
opt.id = '_g@'..opt.game_session..'_h@'..opt.hidden_size..'_d@'..opt.dropout..'_f@'..opt.feat_size..'_a@'..opt.vocab_size


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
	--opt.input_h5_images = '../DATA/game/v1/ALL_REFERIT.h5'
        opt.input_h5_images = '../DATA/game/v1/test.h5'
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

------------------------------------------------------------------------------
-- Printing opt
-----------------------------------------------------------------------------
print(opt)

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

  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train'}
