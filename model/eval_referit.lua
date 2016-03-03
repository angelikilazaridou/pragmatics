require 'torch'
require 'nn'
require 'nngraph'
require 'misc.LinearNB'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoaderRaw'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluating discriminative model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','','path to model to evaluate')
-- Basic options
cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', 100, 'how many images to use when periodically evaluating the loss? (-1 = all)')
-- For evaluation on the Carina images from some split:
cmd:option('-input_h5','/home/angeliki/git/pragmatics/DATA/referit/test_data/data_referit_','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/home/angeliki/git/pragmatics/DATA/referit/test_data/data_referit_','path to the json containing the preprocessed dataset')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-threshold',1,'What threshold to use')
cmd:option('-maxSize',20,'Balances dataset')
cmd:option('-keepOne',-1,'Keep only one feature')
cmd:option('-exclude',0,'Whether to evaluate if feature is exlcluded (precision, STRATEGY=3) or included (recall, STRATEGY=1 or2)')
cmd:option('-strategy',1,'What dataset to evaluate on')
-- misc
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)

------------------------------------------------------------------------------
opt.input_h5 = opt.input_h5..opt.strategy..'.h5'
opt.input_json = opt.input_json..opt.strategy..'.json'

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU


if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'vocab_size', 'hidden_size', 'feat_size'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader =  DataLoaderRaw{h5_file = opt.input_h5,  json_file = opt.input_json, feat_size = opt.feat_size, gpu = opt.gpuid, vocab_size = opt.vocab_size, game_size = 2}


-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.crit = nn.MSECriterion()

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, attributes, evalopt)
        local verbose = utils.getopt(evalopt, 'verbose', true)
        local num_images = utils.getopt(evalopt, 'num_images', true)

        protos.model:evaluate()
        loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
        local n = 0
        local loss_sum = 0
        local loss_evals = 0
        local predictions = {}
        local correct = 0
        local all = 0
	local TP_FP = 0

	local i = 0
	local balance, tmp

	if opt.gpuid <0 then
		balance = torch.zeros(loader:getVocabSize())
		tmp = torch.Tensor(1,loader:getVocabSize()):zero()
	else
		balance = torch.CudaTensor(loader:getVocabSize()):fill(0)
		tmp = torch.CudaTensor(1,loader:getVocabSize()):fill(0)
	end

	
        for i=1,loader:getVocabSize() do
                tmp[1][i] = i
        end

	local nn = 0
        while true do

		i=i+1
                -- fetch a batch of data
                local data = loader:getBatch{batch_size = opt.batch_size, split = split}

                -- forward the model to get loss
                local outputs = protos.model:forward({unpack(data.images)})
                local predicted = outputs[1]

		--get in a tensor the properties
                local properties = {unpack(outputs,2)}
		
		local loss

		--if we have labels, compute loss
		if data.labels then
                	loss = protos.crit:forward(predicted, data.labels)
			--average loss
                	loss_sum = loss_sum + loss
                	loss_evals = loss_evals + 1

                	--compute accuracy
                	local predicted2 = predicted:clone()
			if opt.keepOne ==1 then
				max = torch.max(predicted2)
			else 
				max = 0.5
			end
		
			predicted2:apply(function(x) if x<max then return 0 else return 1 end end)
		
			for i=1,opt.batch_size do
				local skip = 0
				local id = torch.sum(torch.cmul(data.labels[i],tmp))
				--due to some problem, the id can be 0
				if opt.maxSize >0 and id>0 then
					if balance[id]<opt.maxSize then
						balance[id] = balance[id]+1
					else
						skip = 1
					end
				end
				if skip==0 and id>0 then
                        		--check if prediction of discriminativeness if correct
                        		if opt.exclude == 0 and torch.sum(torch.cmul(predicted2[i], data.labels[i]))==1 then
                                		correct = correct + 1
                        		end
					if opt.exclude == 1 and torch.sum(torch.cmul(predicted2[i], data.labels[i]))==0 then
                                                correct = correct + 1
                                        end
					if torch.sum(predicted2)>0 then
						print("#######")
						print(string.format('%s %s %s',data.infos[1].concepts[1], data.infos[1].concepts[2],attributes[id]))
						for p=1,opt.vocab_size do
							if predicted2[1][p]==1 then
								print(attributes[p])
							end
						end
					end
					TP_FP =  TP_FP + torch.sum(predicted2[i])
                        		--check if properties are correct
                        		for ii=1,checkpoint.opt.game_size do
						local s1 = properties[ii][{{i,i},{1,opt.vocab_size}}]:clone():apply(function(x)  if x < opt.threshold  then return 1 else return 0 end end)
                        		end
                        		all = all+1
				end
                	end
		end

                -- if we wrapped around the split or used up val imgs budget then bail
                local ix0 = data.bounds.it_pos_now
                local ix1 = math.min(data.bounds.it_max, num_images)
		
		if data.bounds.wrapped then break end -- the split ran out of data, lets break out
                if loss_evals % 10 == 0 then collectgarbage() end
	end
 
   print('Evaluated on '..all) 
   return correct/all,TP_FP/all
end


local attributes = {}
property_file = '/home/angeliki/git/pragmatics/DATA/visAttCarina/raw/properties.txt'
f = io.open(property_file,'r')
i=1
while true do
        line = f:read()
        if line==null then
                break
        end
        attributes[i] = line:split("%s")[1]
        i = i+1
end

--print(string.format('Params: vocab_size = %d',opt.vocab_size))
local acc, sparsity = eval_split(opt.split, attributes, {num_images = opt.num_images})

print(string.format('Accuracy=%f Sparsity=%f', acc,sparsity))


if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('vis/vis.json', split_predictions)
end
