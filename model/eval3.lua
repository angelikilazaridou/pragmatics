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
cmd:option('-game','v2','Which game to play (v1=REFERIT, v2=OBJECTS, v3=SHAPES). If left empty, json/h5/images.h5 should be given seperately')
cmd:option('-batch_size',1,'what is the batch size of games')
-- Optimization: for the model
cmd:option('-temperature',1,'Temperature') -- tried with 0.5, didn't do the job
-- misc
cmd:option('-val_images_use', 1000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-split','val','What split to use to evaluate')
cmd:option('-print_info',0,'Whether to print info')
cmd:option('-noise',0,'Whwether to you use noise in the input of the Receiver')
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
-- Printing opt
-----------------------------------------------------------------------------
--print(opt)
print("#####################  WORKING WITH FILE ########################")
print(opt.model)
local tt = {}
for t in string.gmatch(opt.model,"[^/]+") do
	table.insert(tt,t)
end
model_name = tt[#tt]
-------------------------------------------------------------------------------
-- Initialize the network
-------------------------------------------------------------------------------
local checkpoint = torch.load(opt.model)
local protos ={}
protos.communication = {}
protos.communication.players = checkpoint.protos


------------------------------------------------------------------------------
-- Input data
------------------------------------------------------------------------------

if opt.game == 'v1' then
        opt.input_json = '../DATA/game/v1/data.json'
        opt.input_h5 = '../DATA/game/v1/data.h5'
        opt.input_h5_images = '../DATA/game/v1/vectors_transposed.h5'
elseif opt.game == 'v2' then
        opt.input_json = '../DATA/game/v2/data.json.new'
        opt.input_h5 = '../DATA/game/v2/data.h5.new'
        if checkpoint.opt.comm_layer == 'probs' then
                opt.input_h5_images = '../DATA/game/v2/images_single.normprobs.h5'
                if checkpoint.opt.comm_viewpoints == 1 then
                        opt.input_h5_images_r = opt.input_h5_images
                else
                        opt.input_h5_images_r = '../DATA/game/v2/vectors_transposed.h5'
                end
        else
                opt.input_h5_images = '../DATA/game/v2/vectors_transposed.h5'
                if checkpoint.opt.comm_viewpoints == 1 then
                        opt.input_h5_images_r = opt.input_h5_images
                else
                        opt.input_h5_images_r = '../DATA/game/v2/images_single.normprobs.h5'
                end
        end
elseif opt.game == 'v3' then
        opt.input_json = '../DATA/game/v3/data.json'
        opt.input_h5 = '../DATA/game/v3/data.h5'
        opt.input_h5_images = '../DATA/game/v3/toy_images.h5'
else
        print('No specific game. Data will be given by user')
end


-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
if checkpoint.opt.layer == "probs" then
	opt.norm = 0
else
	opt.norm = 1
end

local loader = DataLoaderCommunication{h5_file = opt.input_h5, json_file = opt.input_json,  feat_size = checkpoint.opt.feat_size, gpu = checkpoint.opt.gpuid, vocab_size = checkpoint.opt.vocab_size, h5_images_file = opt.input_h5_images, h5_images_file_r = opt.input_h5_images_r, game_size = checkpoint.opt.comm_game_size, embeddings_file_S = "", embeddings_file_R = "", noise = opt.noise, norm = opt.norm, quantize = -1}

opt.vocab_size = checkpoint.opt.vocab_size
vocab_size = opt.vocab_size

labels = csvigo.load({path='../DATA/game/v2/images_single.objects',mode='raw', verbose=false})

obj2id = {}
id2obj = {}
local cnt = 0
for i=1,#labels do
	if obj2id[labels[i][1]]==nil then
		cnt=cnt+1
		obj2id[labels[i][1]] = cnt
		id2obj[cnt]=labels[i][1]
	end	
end

stats = torch.DoubleTensor(cnt,vocab_size):fill(0)
cond = torch.DoubleTensor(cnt,cnt,vocab_size):fill(0)
cond_cor = torch.DoubleTensor(cnt,cnt,vocab_size):fill(0)
items = {}
for a=1,vocab_size do
	items[a] = {}
end
------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
	local verbose = utils.getopt(evalopt, 'verbose', true)
	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

	print(split)
	protos.communication.players:evaluate()
  	
	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local acc = 0
	
	local attribute_usage = torch.DoubleTensor(1,vocab_size):fill(0)
	local correct_attributes = torch.DoubleTensor(1,vocab_size):fill(0)
	local tmp = {}

	all_examples = torch.DoubleTensor(opt.batch_size * math.ceil(opt.val_images_use/opt.batch_size),vocab_size):fill(0)
	local curr_idx = 1

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
		local outputs = protos.communication.players:forward({inputsS, inputsR, opt.temperature})
		
		--prepage gold data	
		local gold = data.referent_position
    	
		
		for k=1,opt.batch_size do
                        if outputs[1][k][gold[k][1]]==1 then
                                acc = acc+1
                        end
			all_examples[curr_idx] = outputs[4][k]
			curr_idx  = curr_idx + 1
		end
		
	
		to_look = 1	

		loss_evals = loss_evals + 1
    		n = n+opt.batch_size

		if loss_evals % 10 == 0 then collectgarbage() end	
		if n >= val_images_use then break end -- we've used enough images	

		for k=1,opt.batch_size do
			local cor = "Wrong"
			for a=1,vocab_size do
				if outputs[3][k][a] == 1 then 
                    
					-- info for symbol usage o target
					stats[obj2id[labels[data.infos[k][to_look]][1]]][a] = stats[obj2id[labels[data.infos[k][to_look]][1]]][a] +1
					
					attribute_usage[1][a] = attribute_usage[1][a] + 1
					if outputs[1][k][gold[k][1]]==1 then
						correct_attributes[1][a] = correct_attributes[1][a] + 1
						cor = "Correct"
					end
					-- item specific results
					target = data.infos[k][1]
					context = data.infos[k][3]
					tmp = {}
					tmp.target = labels[target][1] tmp.context = labels[context][1] tmp.id_1 = target tmp.id_2 = context tmp.answer = cor
					items[a][#items[a]+1] = tmp
                    			if cor == "Correct" then
                        			cond_cor[obj2id[labels[target][1]]][obj2id[labels[context][1]]][a] = cond_cor[obj2id[labels[target][1]]][obj2id[labels[context][1]]][a] +  1
                    			end
                    			cond[obj2id[labels[target][1]]][obj2id[labels[context][1]]][a] = cond[obj2id[labels[target][1]]][obj2id[labels[context][1]]][a] +  1
					break
				end
			end
		end
	end

    	print("######################     ACCURACY   #######################################")
   	print(string.format("Accuracy is %f",acc/n))

    	print("######################    ATTRIBUTE USAGE   ##########################")
   	print(attribute_usage)
    	print("######################   ATTRIBUTE ACCURACY  #########################")
    	print(torch.cdiv(correct_attributes,attribute_usage))
	
	return loss_sum/loss_evals, acc/n

end

loss,acc = eval_split(opt.split, {val_images_use = opt.val_images_use, verbose=opt.verbose})

-- for each target find the most used attribute and its frequency
v,ind = torch.max(stats,2)

clusters = {}
symbols = {}
print("#####################     MOST FREQUENT ATTRIBUTE PER CONCEPT   ############")
-- iterate over concepts
for a=1,stats:size(1) do
	local symbol = ind[a][1]
        local concept = id2obj[a]

	for s=1,vocab_size do
		if symbols[s] == nil then symbols[s] = {} end
		if stats[a][s] > 0 then symbols[s][#symbols[s]+1]= concept end
	end
	-- if frequ attribute is active
	if v[a][1] >0 then
		-- how often is used for that object
		local prop = v[a][1]/torch.sum(stats[a])
		if clusters[symbol] == nil  then
			clusters[symbol] = {}
		end
		print(string.format("Concept %s -- Symbol %d (%f)",concept, symbol, prop))
		clusters[symbol][#clusters[symbol]+1] = concept
	end
end
print("#######################    CLUSTERS OF CONCEPTS     #########################")
print(clusters)
--print(symbols)
to_visualize = "html/"..model_name..'.txt'
print(string.format("#######################    WRITING FILES TO: %s     #########################",to_visualize))
f = io.open(to_visualize,"w")
f:write(model_name..' '..acc.."\n")
to_sample = 30
for a=1,#items do
	for i=1,math.min(to_sample,#items[a]) do
		s = a..' '..items[a][i].target..' '..items[a][i].id_1..' '..items[a][i].context..' '..items[a][i].id_2..' '..items[a][i].answer..'\n'
		f:write(s)
	end
end
f:close(f)


if 1==2 then
print("#######################   PRINT TAREGT-CONTEXT-SYMBOL INFO   ##########################")
f = io.open("context-symbol.info_"..model_name..".txt","w")
lines = 0
for t=1,cnt do
    for c=1,cnt do
        local sum = torch.sum(cond[t][c])
        if sum~=0 then 
            local s = id2obj[t]..' '..id2obj[c]..' '..sum
            for a=1,vocab_size do
                s = s..' '..cond[t][c][a]/sum
            end
            for a=1,vocab_size do
                s = s..' '..cond[t][c][a]/sum
            end
            f:write(s.."\n")
            lines = lines + 1
        end
    end
end
f:close(f)
print(string.format("Wrote %d lines",lines))
end

-- This assumes `t1` is a 2-dimensional tensor!
function tensor2table(t1)
local t2 = {}
for i=1,t1:size(1) do
  t2[i] = {}
  for j=1,t1:size(2) do
    t2[i][j] = t1[i][j]
  end
end
return t2
end

print("#######################   EXAMPLES SCORES   ##########################")
f_name = "redundancy/"..model_name..".txt"
local t = tensor2table(all_examples)
csvigo.save(f_name, t)
--local f = hdf5.open("redundancy/"..model_name..".txt", "w")
--f:write("scores", all_examples)
--f:close()



