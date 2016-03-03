require 'hdf5'
local utils = require 'misc.utils'

local PropertyLoader = {}
PropertyLoader.__index = PropertyLoader

function PropertyLoader:readlists(input_file)
  local i_stream = io.open(input_file, 'r')
  local outputs = {}
  local line = i_stream:read()
  while (line ~= nil) do
    table.insert(outputs, line)
    line = i_stream:read()
  end 
  i_stream:close()
  return outputs
end

function PropertyLoader:tensorize(data)
  for i=1,#data do
    data[i] = data[i]:split("%s")
    for j=1,#data[i] do
      data[i][j] = tonumber(data[i][j])
    end
  end
  data = torch.Tensor(data)
--  if self.gpu then
--    data = data:cuda()
--  end
  return data
end


function PropertyLoader.init(opt)
  local training_file = opt.training_file
  local testing_file = opt.testing_file
  
  local self = {}
  setmetatable(self, PropertyLoader)
  self.gpu = opt.gpu
   
  local test_data = self:readlists(testing_file)
  local train_data = self:readlists(training_file)
  self.visual_num = 4096
  self.prop_num = 576
  self.all_num = self.visual_num + self.prop_num
  self.lengths = {}
  self.lengths["train"] = #train_data
  self.lengths["test"] = #test_data
  test_data = self:tensorize(test_data)
  train_data = self:tensorize(train_data)
  self.data = {}
  self.data["train"] = train_data
  self.data["test"] = test_data
  self.iterators = {}
  self.iterators["train"] = 1
  self.iterators["test"] = 1
  self.wrapped = {}
  self.wrapped["train"] = false
  self.wrapped["test"] = false
  return self
end

function PropertyLoader:resetIterator(split)
    self.iterators[split] = 1
    self.wrapped[split] = false
end

function PropertyLoader:getFeatSize()
    return self.visual_num
end

function PropertyLoader:getVocabSize()
    return self.prop_num
end

--[[
  Split is a string identifier (e.g. train|test)
  Returns a batch of data:
  - X (N,M) containing the images
  - y (N,K) containing the properties 
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function PropertyLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  local split_data = self.data[split]
  local split_iterator = self.iterators[split]
  local split_length = self.lengths[split]
  local batch_data = nil
  if (split_iterator + batch_size - 1) <= split_length then
    batch_data = split_data:sub(split_iterator, split_iterator + batch_size - 1)
  else
    self.wrapped[split] = true
    local end_batch = (split_iterator + batch_size - 2) % split_length + 1
    local batch_data1 = split_data:sub(split_iterator, split_length)
    local batch_data2 = split_data:sub(1, end_batch)
    batch_data = batch_data1:cat(batch_data2, 1)
  end
  local batch_x = batch_data:sub(1, batch_size, 1, self.visual_num)
  local batch_y = batch_data:sub(1, batch_size, self.visual_num + 1, self.all_num)
  
  --print(split .. " iteration: " .. split_iterator ..  ", norm x: " .. batch_x:norm() .. ", norm y: " .. batch_y:norm())
  
  split_iterator = (split_iterator + batch_size - 1) % split_length + 1
  if (split_iterator == 1) then self.wrapped[split] = true end
  self.iterators[split] = split_iterator
  if self.gpu >= 0 then
    return {batch_x:cuda(), batch_y:cuda(), self.wrapped[split]}
  else
    return {batch_x:contiguous(), batch_y:contiguous(), self.wrapped[split]}
  end
end

return PropertyLoader

