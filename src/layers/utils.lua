function linearToConv(layer, nInputPlane, nOutputPlane, kW, kH, lib)
   lib = lib or nn
   -- number of filters in the fully-connected layer
   assert(nOutputPlane == layer.bias:size(1))
   -- create new convolution layer
   local convolutionLayer = lib.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH)
   convolutionLayer.bias = layer.bias:clone()
   convolutionLayer.weight = torch.reshape(layer.weight:clone(), nInputPlane, nOutputPlane, kW, kH)
   return convolutionLayer
end