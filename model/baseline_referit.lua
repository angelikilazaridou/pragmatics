require 'torch'
require 'nn'
require 'nngraph'
require 'csvigo'
-- local imports
local utils = require 'misc.utils'
require 'misc.LinearNB'
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
-- For evaluation on the Carina images from some split:
cmd:option('-input_h5','/home/angeliki/git/pragmatics/DATA/referit/test_data/data_referit_','path to the h5file containing the preprocessed dataset')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-threshold',1,'What threshold to use')
cmd:option('-maxSize',20,'Balanced test set')
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
opt.input_json = opt.input_h5..opt.strategy..'.json'
opt.input_h5 = opt.input_h5..opt.strategy..'.h5'

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
local function eval_split(split, evalopt,properties)
        local verbose = utils.getopt(evalopt, 'verbose', true)
        local num_images = utils.getopt(evalopt, 'num_images', true)

	local balance = torch.zeros(loader:getVocabSize())

	local tmp = torch.Tensor(1,loader:getVocabSize()):zero():cuda()
	for i=1,tmp:size(2) do
		tmp[1][i] = i
	end

        protos.model:evaluate()
        loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
        local n = 0
        local loss_sum = 0
        local loss_evals = 0
        local predictions = {}
        local prob = 0
	local all = 0


        while true do

		i=i+1
                -- fetch a batch of data
                local data = loader:getBatch{batch_size = opt.batch_size, split = split}


		--if we have labels, compute loss
		if data.labels then
			for i=1,opt.batch_size do
				local skip = 0
                                local id = torch.sum(torch.cmul(data.labels[i],tmp))
                                if opt.maxSize >0 and id>0 and balance[id]<opt.maxSize  then
                                        balance[id] = balance[id]+1
                                else
                                        skip = 1
                                end
                                if skip==0 then
					if opt.exclude == 1 then
						prob = prob + (properties[id]^2 + (1-properties[id])^2)
					else
                        			--check if prediction of discriminativeness if correct
						prob = prob+(2*properties[id]*(1-properties[id]))
					end
			
                        		all = all+1
				end
                	end
		end

                -- if we wrapped around the split or used up val imgs budget then bail
                local ix0 = data.bounds.it_pos_now
		
		if data.bounds.wrapped then break end -- the split ran out of data, lets break out
                if loss_evals % 10 == 0 then collectgarbage() end
	end

   print('Evaluated on '..all..' data')
   return prob/all
end

properties = torch.zeros(opt.vocab_size)
property_file = '/home/angeliki/git/pragmatics/DATA/visAttCarina/raw/properties.txt'
f = io.open(property_file,'r')
i=1
local penalty = 0
while true do
	line = f:read()
	if line==null then
		break
	end	
	properties[i] = tonumber(line:split("%s")[2]/462)
	penalty = penalty + (2*properties[i]*(1-properties[i]))
	i = i+1
end
print(properties[1])
print(penalty)
print(string.format('Params: vocab_size = %d',opt.vocab_size))
local acc = eval_split(opt.split, {num_images = opt.num_images}, properties)

print(string.format('Accuracy=%f', acc))


if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('vis/vis.json', split_predictions)
end
