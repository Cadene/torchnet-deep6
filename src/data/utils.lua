
function iterOverDataset(dataset, maxiter, hook)
   for sample in dataset:iterator()() do
      if hook.onSample then hook.onSample(sample) end
      xlua.progress(i, maxiter)
      i = i + 1
      if i >= maxiter then break end
   end
   if hook.onEnd then hook.onEnd() end
end

function processMeanStd(dataset, pc, mean, std)
   require 'xlua'
   local i = 0
   local maxiter = torch.round(dataset:size() * pc)
   print('Process mean')
   iterOverDataset(dataset, maxiter, {
      onSample = function(sample) mean:add(sample.input) end,
      onEnd = function() mean:div(maxiter) end
   })
   print('Process std')
   iterOverDataset(dataset, maxiter, {
      onSample = function(sample) std:add((sample.input - mean):pow(2)) end,
      onEnd = function() std:div(maxiter) end
   })
   return mean, std
end
