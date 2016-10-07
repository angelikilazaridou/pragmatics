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
cmd:option('-temperature',1,'Temperature') -- tried with 0.5, didn't do the job
-- misc
cmd:option('-val_images_use', 1000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-split','val','What split to use to evaluate')
cmd:option('-print_info',0,'Whether to print info')
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
        opt.input_h5_images = '../DATA/game/v1/vectors_transposed.h5'
elseif opt.game_session == 'v2' then
        opt.input_json = '../DATA/game/v2/data.json.new'
        opt.input_h5 = '../DATA/game/v2/data.h5.new'
        opt.input_h5_images = '../DATA/game/v2/images_single.normprobs.h5'
elseif opt.game_session == 'v3' then
        opt.input_json = '../DATA/game/v3/data.json'
        opt.input_h5 = '../DATA/game/v3/data.h5'
        opt.input_h5_images = '../DATA/game/v3/toy_images.h5'
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
local protos ={}
protos.communication = {}
protos.communication.players = checkpoint.protos


-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoaderCommunication{h5_file = opt.input_h5, json_file = opt.input_json,  feat_size = checkpoint.opt.feat_size, gpu = checkpoint.opt.gpuid, vocab_size = checkpoint.opt.vocab_size, h5_images_file = opt.input_h5_images, h5_images_file_r = opt.input_h5_images, game_size = checkpoint.opt.game_size, embeddings_file_S = "", embeddings_file_R = "", noise = 0}

opt.vocab_size = checkpoint.opt.vocab_size


-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
	local verbose = utils.getopt(evalopt, 'verbose', true)
	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

	protos.communication.players:evaluate()
  	
	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local acc = 0
	while true do

		-- get batch of data  
		local data = loader:getBatch{batch_size = opt.batch_size, split = 'val'}

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

		--[[for b=1,data.discriminativeness:size(1) do
			local k = 30
			if data.discriminativeness[b][k]==1 then
				print(torch.sum(torch.cmul(outputs[3][b],idx)))
			end
		end]]--
		
	
		--prepage gold data	
		local gold = data.referent_position
    	
		
		for k=1,opt.batch_size do
                        if outputs[1][k][gold[k][1]]==1 then
                                acc = acc+1
                        end
		end
		
			
	
		-- if we wrapped around the split or used up val imgs budget then bail
		local ix0 = data.bounds.it_pos_now
		local ix1 = math.min(data.bounds.it_max, val_images_use)	

		loss_evals = loss_evals + 1
    		n = n+opt.batch_size

		if loss_evals % 10 == 0 then collectgarbage() end	
		if n >= val_images_use then break end -- we've used enough images	

		if opt.print_info==1 then
			local correct_answers = torch.CudaTensor(1,opt.batch_size):fill(0)
			local correct_attributes = torch.CudaTensor(opt.batch_size, outputs[3]:size(2)):fill(0)
			for k=1,opt.batch_size do
				if outputs[1][k][gold[k][1]]==1 then
					correct_answers[1][k] = 1
				end
				for a=1,outputs[3]:size(2) do
					if outputs[3][k][a] == 1 and correct_answers[1][k] == 1 then
						correct_attributes[k][a] = correct_attributes[k][a] +1
					end
				end

			end
			to_print = torch.random(100)
			if to_print%10 == 0 then
                		print(torch.cdiv(torch.sum(correct_attributes,1),torch.sum(outputs[3],1)))
				print(torch.sum(outputs[3],1))
        		end
		end

	end

	return loss_sum/loss_evals, acc/n

end

loss,acc = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})

print(acc)
loss,acc = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})
print(acc)
loss,acc = eval_split('val', {val_images_use = opt.val_images_use, verbose=opt.verbose})
print(acc)

