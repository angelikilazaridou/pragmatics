require 'dp'


local oracle, parent = torch.class('nn.Oracle_P2', 'nn.Module')


function oracle:__init(batch_size, vocab_size)
	parent.__init(self)

	self.batch_size = batch_size
	self.vocab_size = vocab_size


end


--called from oracle:forward()
function oracle:updateOutput(input)

	-- input has:
	--1 2, : the features for the two images
	local ref1 = input[1]
	local ref2 = input[2]
	--3: the sampled feature
	local feat = input[3]
	local predictions = torch.CudaTensor(self.batch_size,2):fill(0)

	local p1  = torch.sum(torch.cmul(ref1,feat),2)
	local p2 = torch.sum(torch.cmul(ref2,feat),2)
	local changed = false
	for v=1,self.batch_size do
		changed = false
		if p1[v][1]== 1 then
			if p2[v][1] == 0 then
				predictions[v][1] = 1
				changed = true
			end	
		end
		if p2[v][1] == 1 then
			if p1[v][1] == 0 then
				predictions[v][2] = 1
				changed = true
			end
		end
		if changed == false then
			predictions[v][torch.random(2)] = 1
		end
	end

	return predictions
end




