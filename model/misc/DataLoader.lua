require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
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
  	self.num_images = images_size[1]
	if opt.feat_size == -1 then
		self.feat_size = images_size[2]
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
  	for i=1,self.game_size*2 do -- p1 always has referent on position 1, shuffle for p2
		if self.gpu<0 then
    			table.insert(img_batch, torch.FloatTensor(batch_size,  self.feat_size))
		else
			table.insert(img_batch, torch.CudaTensor(batch_size,  self.feat_size))
		end
  	end

	--the labels per game
	local label_batch, properties
	if self.gpu<0 then
		label_batch =  torch.FloatTensor(batch_size, 1)
		discriminativeness = torch.FloatTensor(batch_size, self.vocab_size)
	else
		label_batch = torch.CudaTensor(batch_size, 1)
		discriminativeness = torch.CudaTensor(batch_size, self.vocab_size)
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
			local bb
			if ii==1 then
				bb = self.info.refs[ix].bb1_i
			else
				bb = self.info.refs[ix].bb2_i
			end
    			local img = self.h5_file:read('/images'):partial({bb,bb},{1,self.feat_size})
			local img_norm = torch.norm(img)
	       		img = img/img_norm
    			img_batch[ii][i] = img
			
		end
		--fetch discriminativeness 
		local discr = self.h5_file:read('/labels'):partial({ix,ix},{1,self.max_size})
		for j=1,self.max_size do
			--convert sparse to dense format
			local k = discr[1][j]
			if k==0 then
				break
			end
			discriminativeness[i][k]= 1
		end
    	
		local referent_position = torch.random(self.game_size)  --- pick where to place the referent 
		--labels
    		label_batch[{ {i,i} }] = referent_position
		img_batch[ii][3] = img_batch[ii][(((referent_position%2)+1)%2)+1] --if ref == 1 -> 1 else 2
		img_batch[ii][4] = img_batch[ii][(((referent_position%2)+2)%2)+1] --if ref == 1 -> 2 else 1

  	end
	data.discriminativeness = discriminativeness
  	data.images = img_batch
	data.labels = label_batch:contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  	return data
end

