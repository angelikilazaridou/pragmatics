local LogisticRegressionBasline = {}

require 'nn'
require 'nngraph'

function LogisticRegressionBasline.create_model(game_size, feat_size, vocab_size, hidden_size, gpu)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local concat_input = nn.JoinTable(2)(inputs)
  local hidden = nn.Linear(game_size * feat_size, hidden_size)(concat_input)
  hidden = nn.Sigmoid()(hidden)
  local l_out = nn.Linear(hidden_size, vocab_size)(hidden)
  --l_out = nn.Sigmoid()(l_out)
  
  local outputs = {}
  table.insert(outputs, l_out)
  
  local final_module = nn.gModule(inputs, outputs)
  if gpu then
    final_module:cuda()
  end
  return final_module
end 

return LogisticRegressionBasline