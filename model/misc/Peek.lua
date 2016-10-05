local Peek, parent = torch.class('nn.Peek', 'nn.Module')

function Peek:__init(report_func)
   parent.__init(self)
   self.report_func = report_func
end

function Peek:updateOutput(input)

    self.output=input
    if self.report_func then
      print(self.report_func(input))
    else
      to_print = torch.random(1000)
        if to_print%100 == 0 then
                print(input[1])
        end
    end

    return self.output
end

function Peek:updateGradInput(input, gradOutput)
   -- print('Module grad output')
    --print(gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end
