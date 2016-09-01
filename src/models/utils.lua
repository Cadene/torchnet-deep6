require 'src.layers.ConstAffine'

local utils = {}

-- from https://github.com/szagoruyko/imagine-nn/blob/master/utils.lua
-- ex: BNtoFixed(net, false)
function utils.BNtoFixed(net, ip)
  local net = net:replace(function(x)
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

return utils
