require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:readlists(input_file)
  local i_stream = io.open(input_file, 'r')
  local outputs = {}
  local line = i_stream:read()
  while (line ~= nil) do
    table.insert(outputs, line)
  end 
  i_stream:close()
  return outputs
end

function DataLoader:tensorize(data)
  for i=1,#data do
    data[i] = data[i]:split("%s")
    for j=1,#data[i] do
      data[i][j] = tonumber(data[i][j])
    end
  end
  data = torch.Tensor(data)
  return data
end

function DataLoader:__init(opt)
  local training_file = opt.training_file
  local testing_file = opt.testing_file
  local visual_num = 4096
  local property_num = 576
  local test_data = self:readlists(testing_file)
  local train_data = self:readlists(training_file)
  test_data = self:tensorize(test_data)
  train_data = self:tensorize(train_data)
  self.train_data = train_data
  self.test_data = test_data
  return self
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
      --fetch their properies
      local inx = self.info.images[ix].concepts[ii]
      properties[i][ii] = self.h5_file:read('/properties'):partial({inx,inx},{1,self.vocab_size})
    end
     
    --labels
    seq = torch.FloatTensor(self.feat_size)
    seq = self.h5_file:read('/labels'):partial({ix,ix},{1,self.vocab_size}) 
        label_batch[{ {i,i} }] = seq

        -- and record associated info as well
        local info_struct = {}
        info_struct.id = self.info.images[ix].id
    info_struct.concepts = self.info.images[ix].concepts
        table.insert(infos, info_struct)
    end
  
  --local data = {}
  data.properties = properties
    data.images = img_batch
  data.labels = label_batch:contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
    data.infos = infos
    return data
end

