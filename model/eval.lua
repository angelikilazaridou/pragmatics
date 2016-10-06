require 'torch'
require 'rnn'
require 'dpnn'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoaderCommunication'
require 'misc.optim_updates'
require 'players.Players'
require 'gnuplot'
require 'csvigo'
require 'klib.lua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Symbol analysis')
cmd:text()
cmd:text('Options')

-- Input model
cmd:option('-model','','path to model to evaluate')
-- Basic options
cmd:option('-game_session','v2','Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-batch_size',1,'what is the batch size of games')
-- Optimization: for the model
cmd:option('-temperature',1,'Initial temperature 2') -- tried with 0.5, didn't do the job
-- misc
cmd:option('-val_images_use', 1000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-split','val','What split to use to evaluate')
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
        opt.input_h5_images = '..DATA/game/v1/vectors_transposed.h5'
elseif opt.game_session == 'v2' then
        opt.input_json = '../DATA/game/v2/data.json.new'
        opt.input_h5 = '../DATA/game/v2/data.h5.new'
        opt.input_h5_images = '..DATA/game/v2/vectors_transposed.h5'
elseif opt.game_session == 'v3' then
        opt.input_json = '../DATA/game/v3/data.json'
        opt.input_h5 = '../DATA/game/v3/data.h5'
        opt.input_h5_images = '..DATA/game/v3/toy_images.h5'
else
        print('No specific game. Data will be given by user')
end

------------------------------------------------------------------------------
-- Printing opt
-----------------------------------------------------------------------------
print(opt)


-------------------------------------------------------------------------------
-- Initialize the network
-------------------------------------------------------------------------------
local checkpoint = torch.load(opt.model)
local players = checkpoint.protos.communication


-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoaderCommunication{h5_file = opt.input_h5, json_file = opt.input_json, feat_size = checkpoint.opt.comm_feat_size, gpu = checkpoint.opt.gpuid, h5_images_file_r = opt.input_h5_images, h5_images_file = opt.input_h5_images, game_size = checkpoint.opt.comm_game_size, feat_size = checkpoint.opt.comm_feat_size, vocab_size = checkpoint.opt.vocab_size, embeddings_file_R = "", embeddings_file_S = ""}

opt.vocab_size = checkpoint.opt.vocab_size

collectgarbage() 

------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', 1000)


  players:evaluate() 
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  	
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local acc = 0

  --keep the count of predicted attributes  
  local predicted = torch.DoubleTensor(opt.vocab_size):fill(0)
 
  local matrix = torch.DoubleTensor(loader:getRealVocabSize(),opt.vocab_size):fill(0)
 
  referent_attrs = {}
  context_attrs = {}
  keyset = {} 
	
	while true do

  		-- get batch of data  
    		local data = loader:getBatch{batch_size = opt.batch_size, split = split}


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
    		local outputs = players:forward({inputsS, inputsR, 1})
   
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
				
					if referent_attrs[data.infos[k].bb1] == nil then
						referent_attrs[data.infos[k].bb1] = torch.DoubleTensor(opt.vocab_size):fill(0)
						context_attrs[data.infos[k].bb1] = torch.DoubleTensor(opt.vocab_size):fill(0)
						keyset[#keyset+1]=data.infos[k].bb1

					end
					if context_attrs[data.infos[k].bb2] == nil then
                                                referent_attrs[data.infos[k].bb2] = torch.DoubleTensor(opt.vocab_size):fill(0)
                                                context_attrs[data.infos[k].bb2] = torch.DoubleTensor(opt.vocab_size):fill(0)
						keyset[#keyset+1]=data.infos[k].bb2
                                        end
					
					referent_attrs[data.infos[k].bb1][predicted_attribute] = referent_attrs[data.infos[k].bb1][predicted_attribute]+1
					context_attrs[data.infos[k].bb2][predicted_attribute]  = context_attrs[data.infos[k].bb2][predicted_attribute]+1
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

    
    		if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    		if n >= val_images_use then break end -- we've used enough images
	end

       return acc/(loss_evals * opt.batch_size),predicted, matrix, vocab

end

acc, predicted, matrix,vocab = eval_split(opt.split, {val_images_use = opt.val_images_use, verbose=opt.verbose})
print(acc)

