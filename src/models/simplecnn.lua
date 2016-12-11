function simpleCnn(nclasses, complexity)
   local c = complexity or 1
   local net = nn.Sequential()
   net:add(nn.SpatialConvolution(1,32*c,5,5))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(2,2,2,2))
   net:add(nn.SpatialBatchNormalization(32*c))
   net:add(nn.SpatialConvolution(32*c,64*c,5,5))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(2,2,2,2))
   net:add(nn.SpatialBatchNormalization(64*c))
   net:add(nn.SpatialConvolution(64*c,128*c,5,5))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialBatchNormalization(128*c))
   net:add(nn.View(-1,128*c*18*18))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(128*c*18*18,1024*c))
   net:add(nn.ReLU(true))
   net:add(nn.BatchNormalization(1024*c))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(1024*c,nclasses))
   return net
end


