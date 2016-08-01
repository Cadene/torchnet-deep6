local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
require 'os'
require 'optim'
ffi = require 'ffi'
require 'image'
local lsplit = string.split

require 'src.data.multiclassloader'
-- require 'src.data.chahantapuscritloader'
require 'src.models.simplecnn'

local cmd = torch.CmdLine()
cmd:option('-usegpu', true, 'use gpu for training')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

torch.manualSeed(1337)

local path = '/net/big/cadene/doc/Deep6Framework2'
local pathdata = path..'/data/interim/Chahan'
local pathdataraw = path..'/data/raw/Chahan'
-- dataProcess(pathdata, pathdataraw)
local trainvalset, classes, class2target = loadDataset(pathdata, 'train')
local testset, _, _ = loadDataset(pathdata, 'test')
print(classes)

local function addTransforms(dataset, mean, std)
   local tnti = vision.TransformImage()
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.path,'/')
      sample.label  = spl[#spl-1]
      sample.target = class2target[sample.label]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         tnti:randomScale{minSize=100,maxSize=120},
         tnti:randomCrop(100),
         tnti:colorNormalize(mean, std)
      }(sample.path)
      return sample
   end)
   return dataset
end

local mean, std
-- add mean and std pointers to nil values
-- so colorNormalize return identity
trainvalset = addTransforms(trainvalset, mean, std)
testset     = addTransforms(testset, mean, std)

trainvalset = trainvalset:shuffle()
trainvalset = trainvalset:split{
   train = 0.7,
   val   = 0.3
}
trainvalset:select('train')

mean = torch.zeros(3, 100, 100)
std  = torch.zeros(3, 100, 100)
-- set mean and std so colorNormalize
-- will be fully effective
processMeanStd(trainvalset, 0.05, mean, std)

torch.save('trainvalset.t7', trainvalset)
torch.save('testset.t7', testset)

local function getIterator(modesub)
   -- modesub = {train,val,test}
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = 3,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
         torch.setdefaulttensortype('torch.FloatTensor')
      end,
      closure   = function(threadid)
         local mode = (modesub == 'train' or modesub == 'val')
                        and 'trainval' or 'test'
         local dataset = torch.load(mode..'set.t7')
         if mode == 'trainval' then
            dataset:select(modesub)
         end
         if modesub == 'train' then
            dataset = dataset:batch(30)
         else
            dataset = dataset:batch(60)
         end
         return dataset
      end,
      transform = function(sample)
         sample.target = torch.Tensor(sample.target):view(-1,1)
         return sample
      end
   }
   print('Stats of '..modesub..'set')
   for i, v in pairs(iterator:exec('size')) do
      print(i, v)
   end
   return iterator
end

local net = simpleCnn(#classes):float()
print(net)

local criterion = nn.CrossEntropyCriterion():float()

require 'src.utils'

local meter = {
   avgvm = tnt.AverageValueMeter(),
   confm = tnt.ConfusionMeter{k=#classes},
   timem = tnt.TimeMeter{unit = true},
   clerr = tnt.ClassErrorMeter{topk = {1,5}}
}

local log = {
   train = createLog('train', 'data'),
   val   = createLog('val', 'data'),
   test  = createLog('test', 'data')
}

local engine = tnt.OptimEngine()
engine.hooks.onStart = function(state) resetMeters(meter) end
engine.hooks.onStartEpoch = function(state) -- training only
   engine.epoch = engine.epoch and (engine.epoch + 1) or 1
end
engine.hooks.onForwardCriterion = function(state)
   meter.timem:incUnit()
   meter.avgvm:add(state.criterion.output)
   meter.clerr:add(state.network.output, state.sample.target)
   meter.confm:add(state.network.output, state.sample.target)
   log[engine.mode]:set{
      loss = meter.avgm:value(),
      acc1 = meter.clerr:value{k = 1}
      acc5 = meter.clerr:value{k = 5},
      time = meter.timem:value()
   }
   print(string.format('%s epoch: %i; avg. loss: %2.4f; avg. error: %2.4f',
      engine.mode, engine.epoch, meter.avgvm:value(), meter.clerr:value{k = 1}))
end
engine.hooks.onEnd = function(state)
   print('End of epoch '..engine.epoch..' on '..engine.mode..'set')
   log[engine.mode]:flush()
   print('Confusion Matrix (rows = gt, cols = pred)')
   print(confm:value())
end
if config.usegpu then
   require 'cunn'
   require 'cudnn'
   cudnn.convert(net, cudnn)
   net       = net:cuda()
   criterion = criterion:cuda()
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- Iterator
local trainiter = getIterator('train')
local valiter   = getIterator('val')
local testiter  = getIterator('test')

for epoch = 1, 10 do
   print('Training ...')
   engine.mode = 'train'
   trainiter:exec('resample')
   engine:train{
      network     = net,
      iterator    = trainiter,
      criterion   = criterion,
      optimMethod = optim.adam,
      config      = {
         learningRate = 3e-4
      },
      maxepoch    = 1,
   }
   print('Testing valset ...')
   engine.mode = 'val'
   engine:test{
      network   = net,
      iterator  = valiter,
      criterion = criterion,
   }
   torch.save('net.t7',net:clearState())
end

print('Testing testset ...')
engine.mode = 'test'
engine:test{
   network   = net,
   iterator  = testiter,
   criterion = criterion,
}
