require 'nn'
require 'nngraph'

local PropertyBaseline = {}

function PropertyBaseline.create_model(feat_size, vocab_size, hidden_size, gpu)
  local inputs = {}
  local outputs = {}
  local x = nn.Identity()()
  table.insert(inputs, x)
  local x_size = feat_size
  if hidden_size > 0 then
    x = nn.Linear(feat_size, hidden_size)(x)
    x = nn.Sigmoid()(x)
    x_size = hidden_size    
  end
  local y_hat = nn.Linear(x_size, vocab_size)(x)
  y_hat = nn.Sigmoid()(y_hat)
  table.insert(outputs, y_hat)
  local bl_model = nn.gModule(inputs, outputs)
  if gpu then
     bl_model:cuda()
  end
  return bl_model
end

return PropertyBaseline