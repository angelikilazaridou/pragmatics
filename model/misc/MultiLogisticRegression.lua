local MultiLR, parent = torch.class('nn.MultiLR', 'nn.Criterion')

function MultiLR:__init(weights)
  self.weights = weights or {1,1}
  self.sig_module = nn.Sigmoid()
end

---- s(input): prob class 1
-- 1 - s(input): prob class 0
function MultiLR:updateOutput(input, target)
  self.sigmoid = self.sig_module:forward(input)
  local revert = (target - 1) * -1
  local tmp = (revert - 0.5) * 2
  tmp:cmul(self.sigmoid)
  --return -(torch.log(revert - tmp):sum())
  return 0
end

---- weights[1]: weight class 0
-- weights[2]: weight class 1
function MultiLR:updateGradInput(input, target)
  local base_grad = self.sigmoid - target
  local tmp = (target - 1) * (self.weights[1] * -1) 
  tmp:cmul(base_grad)
  base_grad:cmul(target)
  return base_grad * self.weights[2] + tmp 
end