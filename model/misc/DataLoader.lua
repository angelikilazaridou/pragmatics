require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.vocab = self.info.ix_to_word
	self.gpu = opt.gpu
  	self.vocab_size = self.info.vocab_size 
	self.game_size = opt.game_size
	if opt.vocab_size >0 then
		self.vocab_size = opt.vocab_size
	end
  	print('vocab size is ' .. self.vocab_size)
  
  	-- open the hdf5 file
  	print('DataLoader loading h5 file: ', opt.h5_file)
  	self.h5_file = hdf5.open(opt.h5_file, 'r')
 	--  open the hdf5 images file
	print('DataLoader loading h5 images file: ', opt.h5_images_file)
        self.h5_images_file = hdf5.open(opt.h5_images_file, 'r')
 
  	-- extract image size from dataset
  	local images_size = self.h5_images_file:read('/images'):dataspaceSize()
  	assert(#images_size == 2, '/images should be a 2D tensor')
  	self.num_images = images_size[2]
	if opt.feat_size == -1 then
		self.feat_size = images_size[1]
	else
		self.feat_size = opt.feat_size
	end
    	print(string.format('read %d images of size %d', self.num_images, self.feat_size))

  
  	-- separate out indexes for each of the provided splits
  	self.split_ix = {}
  	self.iterators = {}
  	for i,img in pairs(self.info.refs) do
    		local split = img.split
    		if not self.split_ix[split] then
      			-- initialize new split
      			self.split_ix[split] = {}
      			self.iterators[split] = 1
    		end
    		table.insert(self.split_ix[split], i)
  	end
  
	for k,v in pairs(self.split_ix) do
    		print(string.format('assigned %d images to split %s', #v, k))
  	end
end

function DataLoader:resetIterator(split)
  	self.iterators[split] = 1
end

function DataLoader:getVocabSize()
	return self.vocab_size
end


function DataLoader:getGameSize()
        return self.game_size
end

function DataLoader:getFeatSize()
        return self.feat_size
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,2,M) containing the images
  - y (N,1) containing the feature id 
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
	local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
	local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)

	-- pick an index of the datapoint to load next
  	local split_ix = self.split_ix[split]
  	assert(split_ix, 'split ' .. split .. ' not found.')

  	-- the images
  	local img_batch = {} 
	local refs = {}
  	for i=1,self.game_size do 
		if self.gpu<0 then
    			table.insert(img_batch, torch.FloatTensor(batch_size,  self.feat_size))
			table.insert(refs,torch.FloatTensor(batch_size, self.vocab_size):fill(0))
		else
			table.insert(img_batch, torch.CudaTensor(batch_size,  self.feat_size))
			table.insert(refs,torch.CudaTensor(batch_size, self.vocab_size):fill(0))
		end
  	end

	--the labels per game
	local label_batch
	if self.gpu<0 then
		label_batch =  torch.FloatTensor(batch_size, 1)
		discriminativeness = torch.FloatTensor(batch_size, self.vocab_size):fill(0)
	else
		label_batch = torch.CudaTensor(batch_size, 1)
		discriminativeness = torch.CudaTensor(batch_size, self.vocab_size):fill(0)
	end

	local max_index = #split_ix
	local wrapped = false
	local infos = {}
  
	for i=1,batch_size do

    		local ri = self.iterators[split] -- get next index from iterator
    		local ri_next = ri + 1 -- increment iterator
    		if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    		self.iterators[split] = ri_next
    		ix = split_ix[ri]
    		assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

		--image representations
		for ii=1,self.game_size do
    			-- fetch bounding box id of images
			local bb
			if ii==1 then
				bb = self.info.refs[ix].bb1_i
			else
				bb = self.info.refs[ix].bb2_i
			end
			--fetch respective image -- NOTE: transposed images
    			local img = self.h5_images_file:read('/images'):partial({1,self.feat_size},{bb,bb}):transpose(1,2)
			--normalize to unit norm
			local img_norm = torch.norm(img)
	       		img = img/img_norm
			--finally store image
    			img_batch[ii][i] = img
		end

		--fetch discriminativeness 
		discriminativeness[i] = self.h5_file:read('/labels'):partial({ix,ix},{1,self.vocab_size})
    	
		--[[	
		local d = 0
		for k=1,self.vocab_size do
			if discr[1][k] == 1 then
				print(string.format('R:%s vs C:%s have %s',self.info.refs[ix].bb1,self.info.refs[ix].bb2,self.vocab[tostring(k)]))
			end
		end
		]]--

		----  create data for P2
		local referent_position = torch.random(self.game_size)  --- pick where to place the referent 
		---- labels
    		label_batch[{ {i,i} }] = referent_position
		---- assign shuffled data for P2
		--local pos1 = (((referent_position%2)+1)%2)+1  --if ref == 1 -> 1 else 2
		--local pos2 = (((referent_position%2)+2)%2)+1  --if ref == 1 -> 2 else 1
		refs[((referent_position+1)%2)+1][i] = self.h5_file:read('/refs'):partial({ix,ix},{1,1},{1,self.vocab_size})
		refs[((referent_position+2)%2)+1][i] = self.h5_file:read('/refs'):partial({ix,ix},{2,2},{1,self.vocab_size})

  	end
	data.discriminativeness = discriminativeness
  	data.images = img_batch
	data.refs = refs
	data.referent_position = label_batch:contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  	return data
end

