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
cmd:text('Attribute analysis')
cmd:text()
cmd:text('Options')

-- Input model
cmd:option('-model','','path to model to evaluate')
-- Basic options
cmd:option('-game_session','','Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-game_size',2,'Number of images in the game')
cmd:option('-batch_size',1,'what is the batch size of games')
cmd:option('-feat_size',1,'The number of image features')
cmd:option('-vocab_size',1,'The number of properties')
-- Optimization: for the model
cmd:option('-temperature',0.00000001,'Initial temperature')
cmd:option('-temperature2',1,'Initial temperature 2') -- tried with 0.5, didn't do the job
-- misc
cmd:option('-val_images_use', 1000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-verbose',false,'How much info to give')
cmd:text()


------------------------------------------------------------------------------
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

local vocab = loader:getVocab()
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
local checkpoint = torch.load(opt.model)
local protos = checkpoint.protos

print(string.format('Parameters are game_size=%d feat_size=%d, vocab_size=%d\n', game_size, feat_size,vocab_size))
protos.players = protos.model

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
  local predicted = torch.DoubleTensor(opt.vocab_size):fill(0)
  local matrix = torch.DoubleTensor(loader:getRealVocabSize(),opt.vocab_size):fill(0)
	
	while true do

  		-- get batch of data  
    		local data = loader:getBatch{batch_size = opt.batch_size, split = 'val'}


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
   
    		--prepage gold data
    		local gold = data.referent_position
    

    		for k=1,opt.batch_size do
			--accuracy
      			if outputs[1][k][gold[k][1]]==1 then
        			acc = acc+1
      			end
			local predicted_attribute
			--count predicted attributes
			for a=1,opt.vocab_size do
				if outputs[3][k][a] == 1 then
					predicted_attribute = a
					predicted[a] = predicted[a]+1
					--print(string.format('%s -- %s -- %d',data.infos[k].bb1,data.infos[k].bb2,a))
					break
				end
			end
			--assign attributes to features
			for j=1,matrix:size(1) do
				if data.discriminativeness[k][j] == 1 then
					matrix[j][predicted_attribute] = matrix[j][predicted_attribute] +1
				end	
			end
			
    		end

 		n = n+opt.batch_size
		loss_evals = loss_evals+1
	
    		-- if we wrapped around the split or used up val imgs budget then bail
    		local ix0 = data.bounds.it_pos_now
    		local ix1 = math.min(data.bounds.it_max, val_images_use)

    
		if loss_evals % 10 == 0 then collectgarbage() end
    		if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    		if n >= val_images_use then break end -- we've used enough images
	end

       return acc/(loss_evals * opt.batch_size),predicted, matrix, vocab

end

acc, predicted, matrix,vocab = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})

print(acc)

print(predicted)
--print(matrix)
    

-- if game = 2, reorder
if opt.game_session == 'v2' then

	concepts_f = 'scripts/concepts_cats.txt'
	f = io.open(concepts_f,'r')
	mapping = {}	
	i = 1
	while true do
        	line = f:read()
        	if line==null then
                	break
        	end
		local c  = line:split("%s")[1]
		local found = 0
		for s=1,matrix:size(1) do
			if vocab[tostring(s)] == c then
				mapping[i] = s
				break
			end
		end
		i = i+1
	end
	print(mapping)
end
local r_norm_m = matrix:norm(2,2)
matrix:cdiv(r_norm_m:expandAs(matrix))

-- cosine
local cosine = matrix * matrix:transpose(1,2)

-- ranking
for s = 1,matrix:size(1) do
    vals,index = torch.sort(cosine:select(1,s),true) -- sort rows
--    print(string.format('%s similar to %s (%f)',vocab[tostring(s)],vocab[tostring(index[2])], vals[2])) -- 2 cause 1 is the same
end

