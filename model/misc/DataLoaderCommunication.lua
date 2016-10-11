require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoaderCommunication')

function DataLoader:__init(opt)
  
	-- load the json file which contains additional information about the dataset
	--print('COMMUNICATION: DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	-- TODO: in case we load word embeddings, make sure to find correspondance with old vocabulary
	self.vocab = self.info.ix_to_word
	self.gpu = opt.gpu
	self.noise = opt.noise
	self.norm = opt.norm
	self.real_vocab_size = self.info.vocab_size
	self.game_size = opt.game_size

	self.embeddings = {}

	-- load word embeddings and set vocab size of receiver
	if opt.embeddings_file_R~="" then
		self.embeddings["receiver"] = {}
		emb, idx = self:load_embeddings(opt.embeddings_file_R, opt.embedding_size_R)
		self.embeddings.receiver["matrix"] = emb:clone()
		if self.gpu >=0 then
			self.embeddings.receiver.matrix = self.embeddings.receiver.matrix:cuda()
		end
		self.embeddings.receiver["idx"] = idx
		self.vocab_size = #idx
		self.vocab = idx
	end
	-- load word embeddings and set vocab size of sender
        if opt.embeddings_file_S~="" then
                self.embeddings["sender"] = {}
                emb, idx = self:load_embeddings(opt.embeddings_file_S, opt.embedding_size_S)
                self.embeddings.sender["matrix"] = emb:clone()
                if self.gpu >=0 then
                        self.embeddings.sender.matrix = self.embeddings.sender.matrix:cuda()
                end
                self.embeddings.sender["idx"] = idx
                self.vocab_size = #idx
		self.vocab = idx
        end
	if self.vocab_size==nil then
		if opt.vocab_size >0 then
			self.vocab_size = opt.vocab_size
		else
			self.vocab_size = self.info.vocab_size
		end 	
	end
  	--print('vocab size is ' .. self.vocab_size)
  
  	-- open the hdf5 file
  	--print('COMMUNICATION: DataLoader loading h5 file: ', opt.h5_file)
  	self.h5_file = hdf5.open(opt.h5_file, 'r')
 	--  open the hdf5 images file
	--print('COMMUNICATION: DataLoader loading h5 images file: ', opt.h5_images_file)
	self.h5_images_file = hdf5.open(opt.h5_images_file, 'r')
 	self.h5_images_file_r = hdf5.open(opt.h5_images_file_r, 'r')

  	-- extract image size from dataset
  	self.images_size = self.h5_images_file:read('/images'):dataspaceSize()
    --print(self.images_size)
  	assert(#self.images_size == 2, '/images should be a 2D tensor')
	local feat_size 
	self.num_images = self.images_size[1] -1
	feat_size = self.images_size[2]
	
	if opt.feat_size == -1 then
		self.feat_size = feat_size
	else
		self.feat_size = opt.feat_size
	end
  	--print(string.format('read %d images of size %d', self.num_images, self.feat_size))

  
	labels = csvigo.load({path='../DATA/game/v2/images_single.objects',verbose=false,mode='raw'})

        self.obj2id = {}
        local keys = {}
        local cnt = 0
        for i=1,#labels do
                if keys[labels[i][1]]==nil then
                        cnt = cnt + 1
                        keys[labels[i][1]] = cnt
                        self.obj2id[cnt] = {}
                        self.obj2id[cnt].labels = labels[i][1]
                        self.obj2id[cnt].ims = {}
                end
                self.obj2id[cnt]["ims"][#self.obj2id[cnt]["ims"]+1] = i
        end
	
	
end

function DataLoader:resetIterator(split)
  	self.iterators[split] = 1
end

function DataLoader:getVocabSize()
	return self.vocab_size
end

function DataLoader:getRealVocabSize()
	return self.real_vocab_size
end

function DataLoader:getGameSize()
        return self.game_size
end

function DataLoader:getFeatSize()
        return self.feat_size
end

function DataLoader:getEmbeddings(player)
	return self.embeddings[player]
end

function DataLoader:load_embeddings(a,dims)
 
	local d = csvigo.load({path=a, mode="large", verbose=false})
	local header = d[1][1]:split("[ \t]+")

	local rows = #d
	if dims<0 then
		dims = #header-1
	end

	local idx = {}
	local vecs = torch.FloatTensor(rows,dims):fill(0)
	
	for i=1,rows do
        	local line = d[i][1]:split("[ \t]+")
                vecs[i] = torch.FloatTensor({unpack(line, 2, dims+1)})
		-- normalize to unit norm ALWAYS!
		local n = torch.norm(vecs[i])
		vecs[i] = vecs[i]/n
                idx[i] = line[1]
        end

	print(string.format("Loaded %d embeddings",rows))
	
	return vecs, idx
end


function DataLoader:getVocab()
	return self.vocab
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


  	-- the images
  	local img_batch = {} 
	local refs = {}
  	for i=1,self.game_size do 
    		table.insert(img_batch, torch.FloatTensor(batch_size,  self.feat_size))
		table.insert(refs,torch.FloatTensor(batch_size, self.feat_size):fill(0))
  	end

	--the labels per game
	local discriminativeness = torch.FloatTensor(batch_size, self.real_vocab_size)
	local label_batch =  torch.FloatTensor(batch_size, 1)
	
	local infos = {}
  
  	local i=1
	while i<=batch_size do

		
		local indices = torch.randperm(self.game_size)
	    	label_batch[{ {i,i} }] = indices[1]	

		local info_struct = {}
		local prev_c = -1
		--image representations
		for ii=1,self.game_size do
	
			-- pick a concept               
                        local c = torch.random(#self.obj2id)
			-- same concepts game
                        if split == "test_hard" then
                                if prev_c == -1 then
                                        prev_c = c
                                else
                                        c = prev_c
                                end
                        end

			-- pick an image of that concept for P1
                        local bb1 = self.obj2id[c]["ims"][torch.random(#(self.obj2id[c].ims))]
                        table.insert(info_struct,bb1)
			local img = self.h5_images_file:read('/images'):partial({bb1,bb1}, {1,self.feat_size})
			--normalize to unit norm
			if self.norm == 1 then img_batch[ii][i] = img/torch.norm(img) else img_batch[ii][i] = img end

			-- pick an image of the same concept for P2
                        local bb2 = self.obj2id[c]["ims"][torch.random(#(self.obj2id[c].ims))]
                        while bb2 == bb1 do
                                bb2 = self.obj2id[c]["ims"][torch.random(#(self.obj2id[c].ims))]
			end
			if self.noise == 0 then bb2 = bb1 end
                        table.insert(info_struct,bb2)
			local img =  self.h5_images_file_r:read('/images'):partial({bb2,bb2}, {1,self.feat_size})
			-- normalize to unit notm
                        if self.norm == 1 then refs[indices[ii]][i] = img/torch.norm(img) else refs[indices[ii]][i] = img end

		end

		i = i+1	

		-- meta
		table.insert(infos,info_struct)
	end

	if self.gpu<0 then
	  	data.images = img_batch
		data.refs = refs
		data.referent_position = label_batch:contiguous() -- note: make label sequences go down as columns
	else
		data.images = {}
		data.refs = {}
		
		for i=1,self.game_size do
	                table.insert(data.images, img_batch[i]:cuda())
        	        table.insert(data.refs, refs[i]:cuda())
		end
		
                data.referent_position = label_batch:cuda():contiguous() -- note: make label sequences go down as columns
	end
	data.infos = infos
  	return data
end

