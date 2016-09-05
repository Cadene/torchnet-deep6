require 'src.layers.ConstAffine'

local utils = {}

-- from https://github.com/szagoruyko/imagine-nn/blob/master/utils.lua
-- ex: BNtoFixed(net, false)
function utils.BNtoFixed(net, ip)
   local net = net:replace(
      function(x)
         if torch.typename(x):find'BatchNormalization' then
            local no = x.running_mean:numel()
            local y = ConstAffine(no, ip):type(x.running_mean:type())
            assert(x.running_var and not x.running_std)
            local invstd = x.running_var:double():add(x.eps):pow(-0.5)
            y.a:copy(invstd)
            y.b:copy(-x.running_mean:double():cmul(invstd:double()))
            if x.affine then
               y.a:cmul(x.weight)
               y.b:cmul(x.weight):add(x.bias)
            end
            return y
         else
            return x
         end
      end
   )
   assert(#net:findModules'nn.SpatialBatchNormalization' == 0)
   assert(#net:findModules'nn.BatchNormalization' == 0)
   assert(#net:findModules'cudnn.SpatialBatchNormalization' == 0)
   assert(#net:findModules'cudnn.BatchNormalization' == 0)
   return net
end

function utils.paramsNumber(net)
   local netParams = 0
   local bnParams = 0
   for i, module in ipairs(net:listModules()) do
      local moduleParams = 0
      local w, b = module.weight, module.bias
      local weightParams, biasParams = 0, 0
      if w then
         weightParams = 1
         for j = 1, w:dim() do weightParams = weightParams * w:size(j) end
      end
      if b then
         for j = 1, b:dim() do biasParams = biasParams + b:size(j) end
      end
      moduleParams = weightParams + biasParams
      if torch.typename(module) == 'nn.SpatialBatchNormalization' then
         print(module, moduleParams, '*')
      else
         netParams = netParams + moduleParams
         print(module, moduleParams)
      end
   end
   print('Total', netParams)
end

return utils
