require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
local PropertyLoader = require 'misc.PropertyLoader'
require 'misc.LinearNB'

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
cmd:option('-dump_images', 0, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 0, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-num_images',3200,'How many images to use')
-- For evaluation on a folder of images:
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
-- For evaluation on the Carina images from some split:
cmd:option('-training_file','/home/thenghiapham/work/project/pragmatics/DATA/visAttCarina/processed/prop_baseline/train.txt','path to the h5file containing the preprocessed dataset')
cmd:option('-testing_file','/home/thenghiapham/work/project/pragmatics/DATA/visAttCarina/processed/prop_baseline/test.txt','path to the h5file containing the preprocessed dataset')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: test|train')
cmd:option('-threshold',0.5,'What threshold to use')
cmd:option('-beta',1,'F_beta evaluation')
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
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'vocab_size', 'hidden_size', 'feat_size'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader
if string.len(opt.image_folder) == 0 then
  loader =  PropertyLoader.init{training_file = opt.training_file, testing_file = opt.testing_file, gpu = opt.gpuid}

end

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.crit = nn.MSECriterion()

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
        local verbose = utils.getopt(evalopt, 'verbose', true)
        local num_images = utils.getopt(evalopt, 'num_images', true)

        protos.model:evaluate()
        loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
        local n = 0
        local loss_sum = 0
        local loss_evals = 0
        local predictions = {}
  --- for properties
  local TP = 0
  local TP_FN = 0
  local TP_FP = 0
  --for discriminativeness
  local TP_D = 0
  local TP_FN_D = 0
  local TP_FP_D = 0
  local i = 0

  local sparsity = 0
        while true do

    i=i+1
                -- fetch a batch of data
                local data = loader:getBatch{batch_size = opt.batch_size, split = split}

                -- forward the model to get loss
                local outputs = protos.model:forward(data[1])

    --get in a tensor the properties
    local labels = data[2]
    local loss

    --if we have labels, compute loss
    if labels then
      loss = protos.crit:forward(outputs, labels)
      --average loss
      loss_sum = loss_sum + loss
      loss_evals = loss_evals + 1

      --compute accuracy
      local predicted2 = outputs:clone()
      local threshold = tonumber(evalopt.threshold)
      predicted2:apply(function(x) if x>threshold then return 1 else return 0 end end)
                
                          --check if prediction of discriminativeness if correct
        TP_D = TP_D + torch.sum(torch.cmul(predicted2,labels))
        TP_FN_D = TP_FN_D + torch.sum(labels)
        TP_FP_D = TP_FP_D + torch.sum(predicted2)
        
        
    end

    
    if data[3] then break end -- the split ran out of data, lets break out
    if loss_evals % 10 == 0 then collectgarbage() end
  end
  
  print(TP_FP_D/i)

  local precision_D = TP_D/TP_FP_D
  local recall_D = TP_D/TP_FN_D
  local F1_D = ((1+opt.beta^2) * precision_D * recall_D)/((opt.beta^2*precision_D)+recall_D) 
  
  return precision_D, recall_D, F1_D
end

print(string.format('Params: vocab_size = %d',opt.vocab_size))
local  precision_D, recall_D, F1_D = eval_split(opt.split, {num_images = opt.num_images, threshold = opt.threshold})

print(string.format('Discriminativeness eval: Precision=%f Recall=%f F1=%f', precision_D, recall_D, F1_D))


if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('vis/vis.json', split_predictions)
end
