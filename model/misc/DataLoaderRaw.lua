require 'hdf5'
local utils = require 'misc.utils'

local DataLoaderRaw = torch.class('DataLoaderRaw')

function DataLoaderRaw:__init(opt)


	print('DataLoader loading json file: ', opt.json_file)
        self.info = utils.read_json(opt.json_file)

  	--handcode split 
	split = 'test'
	
	-- load the json file which contains additional information about the dataset
	self.gpu = opt.gpu
  	self.vocab_size = opt.vocab_size 
	if opt.vocab_size >0 then
		self.vocab_size = opt.vocab_size
	end
	self.game_size = opt.game_size
  	print('vocab size is ' .. self.vocab_size)
  
  	-- open the hdf5 file
  	print('DataLoader loading h5 file: ', opt.h5_file)
  	self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  	-- extract image size from dataset
  	local images_size = self.h5_file:read('/images'):dataspaceSize()
  	assert(#images_size == 3, '/images should be a 3D tensor')
  	assert(images_size[2] == 2, 'There should be two images per training element')
  	self.num_images = images_size[1]
	if opt.feat_size == -1 then
		self.feat_size = images_size[3]
	else
		self.feat_size = opt.feat_size
	end
	print(images_size[1])
  	self.num_pairs = images_size[2]
    	print(string.format('read %d images of size %dx%d', self.num_images, self.num_pairs, self.feat_size))

  
  	-- separate out indexes for each of the provided splits
  	self.split_ix = {}
  	self.iterators = {}
  	for i=1,self.num_images do
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

function DataLoaderRaw:resetIterator(split)
  	self.iterators[split] = 1
end

function DataLoaderRaw:getVocabSize()
	return self.vocab_size
end


function DataLoaderRaw:getGameSize()
        return self.game_size
end

function DataLoaderRaw:getFeatSize()
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
function DataLoaderRaw:getBatch(opt)
	local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
	local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
	
  	local split_ix = self.split_ix[split]
  	assert(split_ix, 'split ' .. split .. ' not found.')

  	-- pick an index of the datapoint to load next
  	local img_batch = {} --torch.Tensor(batch_size, mem_size, self.feat_size)
  	--initialize one table elements per game size
  	for i=1,self.game_size do
		if self.gpu<0 then
    			table.insert(img_batch, torch.FloatTensor(batch_size,  self.feat_size))
		else
			table.insert(img_batch, torch.CudaTensor(batch_size,  self.feat_size))
		end
  	end

	--the labels per gane

	local label_batch, properties
	if self.gpu<0 then
		label_batch =  torch.FloatTensor(batch_size, self.vocab_size)
		properties = torch.FloatTensor(batch_size, self.game_size, self.vocab_size)
	else
		label_batch = torch.CudaTensor(batch_size, self.vocab_size)
		properties = torch.CudaTensor(batch_size, self.game_size, self.vocab_size)
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
    			-- fetch the image from h5
    			local img = self.h5_file:read('/images'):partial({ix,ix},{ii,ii},{1,self.feat_size})
			local img_norm = torch.norm(img)
	       		img = img/img_norm	
    			img_batch[ii][i] = img
		end
     
		--labels
		seq = torch.FloatTensor(self.feat_size)
		seq = self.h5_file:read('/labels'):partial({ix,ix},{1,self.vocab_size}) 
    		label_batch[{ {i,i} }] = seq

    		-- and record associated info as well
    		local info_struct = {}
    		info_struct.id = ix
		info_struct.concepts = self.info.concepts[ix]
    		table.insert(infos, info_struct)
  	end
  	data.images = img_batch
	data.labels = label_batch:contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  	data.infos = infos
  	return data
end

