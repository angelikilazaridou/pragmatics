require 'torch'
require 'rnn'
require 'dpnn'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
require 'models.Players'
require 'gnuplot'
require 'csvigo'
require 'klib.lua'

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
        opt.input_json = '../DATA/game/v2/data.json'
        opt.input_h5 = '../DATA/game/v2/data.h5'
        opt.input_h5_images = '..DATA/game/v2/vectors_transposed.h5'
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
-- Define name of file
------------------------------------------------------------------------------

opt.id = 'eval_g@'..opt.game_session..'_h@'..checkpoint.opt.hidden_size..'_d@'..checkpoint.opt.dropout..'_f@'..opt.feat_size..'_a@'..opt.vocab_size
print(string.format("Saving this at %s", opt.id))
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

  --keep the count of predicted attributes  
  local predicted = torch.DoubleTensor(opt.vocab_size):fill(0)
 
  local matrix = torch.DoubleTensor(loader:getRealVocabSize(),opt.vocab_size):fill(0)
 
  referent_attrs = {}
  context_attrs = {}
  keyset = {} 
	
	while true do

  		-- get batch of data  
    		local data = loader:getBatch{batch_size = opt.batch_size, split = split}


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

    
		if loss_evals % 10 == 0 then collectgarbage() end
    		if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    		if n >= val_images_use then break end -- we've used enough images
	end

       return acc/(loss_evals * opt.batch_size),predicted, matrix, vocab

end

acc, predicted, matrix,vocab = eval_split(opt.split, {val_images_use = opt.val_images_use, verbose=opt.verbose})

print(acc)

do_exp1 = 0
do_exp2 = 0
do_exp3 = 0
do_exp4 = 0
do_exp5 = 1
do_exp6 = 1
do_exp7 = 1
-- EXP1: align latent to gold (only SHAPES)
-- see if you can align -- for each latent keep the maximum fitting 
-- (in terms of images) gold attribute. Make sure that 
-- this gold attribute cannot be used again.
if opt.vocab_size == loader:getRealVocabSize() and opt.game_session=='v3' and do_exp1 == 1 then
	--diagonalize values
	new_matrix  = torch.DoubleTensor(matrix:size(1),matrix:size(2)):fill(0)
	old_matrix  = torch.DoubleTensor(matrix:size(1),matrix:size(2))
	old_matrix:copy(matrix:transpose(1,2)) --rows are latent
	
	labels = 'set ytics ('
	for s=1,matrix:size(1) do
		labels = labels .. '\"'.. vocab[tostring(s)]..'\" '..tostring(s-1)..', '
		--maximum fitting gold attribute
		vals, idx = torch.sort(old_matrix[s],true)
		bst_attr = idx[1]
		if vals[1]>0 then vals[1] = 1 end --discretize
		new_matrix[s][s] = vals[1]
		--make sure bst_attr cannot be used again
		old_matrix[{{1,opt.vocab_size},idx[1]}] = 0
	end
	labels = labels..')'
	gnuplot.epsfigure(opt.id..'_attribute_usage.eps')
	if opt.game_session == 'v3' then
		gnuplot.raw(labels)
	end
	gnuplot.imagesc(new_matrix,'color')
	gnuplot.plotflush()
end

-- EXP2: plot pairwise similarities of objects (only OBJECTS)
-- reorder the rows so that they are based on category defined in concepts_cats.txt
-- then plot similarity matrix in the latent attribute vector space
if opt.game_session == 'v2' and do_exp2 ==1 then
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
		for s=1,matrix:size(1) do
			if vocab[tostring(s)] == c then
				mapping[i] = s
				break
			end
		end
		i = i+1
	end
	new_matrix  = torch.DoubleTensor(matrix:size(1), matrix:size(2))

	--normalize for cosine
        local matrix0 = matrix:clone()
	local r_norm_m = matrix0:norm(2,2)
	matrix0:cdiv(r_norm_m:expandAs(matrix0))

	--re-order
	for c=1,matrix:size(1) do
		new_matrix[c] = matrix0[mapping[c]]
	end

	local cosine = matrix0 * matrix0:transpose(1,2)
	
	gnuplot.epsfigure(opt.id..'_shuffles.eps')
	gnuplot.imagesc(cosine,'color')
	gnuplot.plotflush()

	matrix0 = new_matrix
	cosine = matrix0 * matrix0:transpose(1,2)
	gnuplot.epsfigure(opt.id..'_correct.eps')
        gnuplot.imagesc(cosine,'color')
        gnuplot.plotflush()

end

--EXP3: compute the Marco measure of coherence (ALL)
-- ideally, attributes should not be used with images that are both in the context and referent position
inter = torch.DoubleTensor(opt.vocab_size):fill(0)
union = torch.DoubleTensor(opt.vocab_size):fill(0)

active = 0
for a=1,opt.vocab_size do
	for k=1,#keyset do
		img = keyset[k]
		if referent_attrs[img][a]>=1 and context_attrs[img][a]>=1 then -- intersection
			inter[a] = inter[a] + 1
		end
		if referent_attrs[img][a]>=1 or context_attrs[img][a]>=1 then  -- union
                        union[a] = union[a] + 1
                end

	end
	if union[a] > 0 then
		print(string.format("Attribute %d has %f when used %d",a,inter[a]/union[a], predicted[a]))
		active = active + 1
	end
end


--EXP4: compute nearest neighbor of  each annotation in attribute space 
if do_exp4 == 1 then
--normalize for cosine
local r_norm_m = matrix:norm(2,2)
local matrix2 = matrix:clone()
matrix2:cdiv(r_norm_m:expandAs(matrix2))
--compute cosine between annotations
local sims = -matrix2 * matrix2:t()  

sorted, indices = torch.sort(sims, 2)
for f=1,matrix:size(1) do
    print(string.format("[%d] %s --> %s (%f)",f, vocab[tostring(f)],vocab[tostring(indices[f][2])], sorted[f][2]))
end
end


--EXP5: compute attribute similarities
if do_exp5 == 1 then
local annotations_nr = matrix:size(1)
local matrix_t = matrix:t()
local matrix3 = torch.DoubleTensor(active, annotations_nr)

--keep only active attributes to avoid ugly nan
j=1
for i=1,opt.vocab_size do
    if union[i]>=1 then
        matrix3[j] = matrix_t[i]:clone()
        j = j+1
    end
end
local r_norm_m = matrix3:norm(2,2)
matrix3:cdiv(r_norm_m:expandAs(matrix3))
--compute cosine between attributes
local sims = matrix3 * matrix3:t()
gnuplot.epsfigure(opt.id..'_attribute_sims.eps')
gnuplot.imagesc(sims,'color')
gnuplot.plotflush()
end


--EXP6: annotations vs attributes
if do_exp6 == 1 then
gnuplot.epsfigure(opt.id..'_annotations_attributes.eps')
gnuplot.imagesc(matrix,'color')
gnuplot.plotflush()
end

--EXP7: do correlations of annotations and cbow vectors
if do_exp7 == 1 then
    matrix = matrix:double()
    --read cbow vectors
    local embeddings = {}
    local mapping = {}
    local common

    local idx = 1
   
    local suffix = ''
    if opt.game_session == 'v1' then
        suffix = '_subset_top100000'
    else
        suffix = '_subset_v2'
    end
    b = csvigo.load({path="word2vec/cbow"..suffix..".txt", mode="large"})
    rows = csvigo.load({path="word2vec/rows"..suffix..".txt", mode="large"})
    for i=1,#b do
        line = b[i][1]:split("[ \t]+")
        for f=1,matrix:size(1) do
            if vocab[tostring(f)] == rows[i][1] then
                embeddings[idx] = torch.DoubleTensor({unpack(line, 1, #line)})
                mapping[idx] = f
                idx = idx + 1
                --print(string.format("Found %s",vocab[tostring(f)]))
                break
            end
        end
    end
   
    print(#embeddings) 
    -- subset to keep common embeddings 
    local new_annotations = torch.DoubleTensor(#mapping, matrix:size(2))
    local new_embeddings = torch.DoubleTensor(#embeddings, embeddings[1]:size(1))

    for f=1,#embeddings do
        new_embeddings[f] = embeddings[f]:clone()
        new_annotations[f] = matrix[mapping[f]]:clone()
    end

    -- unit norm
    -- new_embeddings = torch.rand(new_embeddings:size(1), new_embeddings:size(2))
    local r_norm_m = new_embeddings:norm(2,2)
    new_embeddings:cdiv(r_norm_m:expandAs(new_embeddings))
    sims1 = new_embeddings * new_embeddings:t()
 
    --new_annotations =  torch.rand(new_annotations:size(1), new_annotations:size(2))
    local r_norm_m = new_annotations:norm(2,2)
    new_annotations:cdiv(r_norm_m:expandAs(new_annotations))
    sims2 = new_annotations * new_annotations:t()

  
    print(new_embeddings:size())
    print(new_annotations:size())
    -- compute pdist :-)
    local to_correlate = {}
    local cur = 1
    for i=1,#embeddings do
        for j=i+1,#embeddings do
            if sims1[i][j]==sims1[i][j] and sims2[i][j]==sims2[i][j] and sims1[i][j]~=0 and sims2[i][j]~=0 then
                to_correlate[cur] = {}
                to_correlate[cur][1] = sims1[i][j]
                to_correlate[cur][2] = sims2[i][j]
                cur = cur + 1
            end
        end
    end
   
  

    print(string.format("(Spearman) correlation between induced attributes and real linguistic ones %f (%d items)",math.spearman(to_correlate),#to_correlate))
    print(string.format("(Pearson) correlation between induced attributes and real linguistic ones %f (%d items)",math.pearson(to_correlate),#to_correlate))

    -- NULL hypothesis
    local max_iterations = 100
    local spearman = 0
    local pearson = 0
    for i=1,max_iterations do
        print(string.format("%d ",i))
        new_embeddings = torch.rand(new_embeddings:size(1), new_embeddings:size(2))
        -- normalize to unit norm
        local r_norm_m = new_embeddings:norm(2,2)
        new_embeddings:cdiv(r_norm_m:expandAs(new_embeddings))

        sims1 = new_embeddings * new_embeddings:t()
        
        -- compute pdist :-)
        local to_correlate = {}
        local cur = 1
        for i=1,#embeddings do
            for j=i+1,#embeddings do
                if sims1[i][j]==sims1[i][j] and sims2[i][j]==sims2[i][j] and sims1[i][j]~=0 and sims2[i][j]~=0 then
                    to_correlate[cur] = {}
                    to_correlate[cur][1] = sims1[i][j]
                    to_correlate[cur][2] = sims2[i][j]
                    cur = cur + 1
               end
            end
        end
    
        spearman = spearman + math.spearman(to_correlate)
        pearson = pearson + math.pearson(to_correlate) 

    end
    print(string.format("NULL: (Spearman) correlation between induced attributes and real linguistic ones %f (%d items)", spearman/max_iterations, #to_correlate))
    print(string.format("NULL: (Pearson) correlation between induced attributes and real linguistic ones %f (%d items)", pearson/max_iterations, #to_correlate))
end
print(string.format("Number of active attributes: %d",active))

