local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
require 'os'
require 'optim'
ffi = require 'ffi'
require 'image'
local lsplit = string.split

require 'src.utils'
require 'src.data.multiclassloader'
require 'src.models.inceptionv3'

local cmd = torch.CmdLine()
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 48, 'batch size')
cmd:option('-nepoch', 50, 'epoch number')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1337)

local path = '/net/big/cadene/doc/Deep6Framework2'
local pathdata = path..'/data/raw/upmcfood101/images'
local pathinceptionv3 = path..'models/inceptionv3/net.t7'
local pathtrainset = path..'data/processed/upmcfood101/trainset.t7'
local pathtestset  = path..'data/processed/upmcfood101/testset.t7'
os.execute('mkdir -p '..paths.dirname(pathtrainset))

local pathlog = path..'logs/upmcfood101/'..os.date("%y_%m_%d_%X")
os.execute('mkdir -p '..pathlog)
local pathtrainlog = pathlog..'/trainlog.txt'
local pathtestlog  = pathlog..'/testlog.txt'
local pathbestepoch = pathlog..'/bestepoch.t7'
local pathbestnet   = pathlog..'/net.t7'

local trainset, classes, class2target = loadDataset(pathdata, 'train')
local testset, _, _ = loadDataset(pathdata, 'test')

local net = vision.models.inceptionv3.load(pathinceptionv3, 10, #classes)
local mean = vision.models.inceptionv3.mean()
local std  = vision.models.inceptionv3.std()
local criterion = nn.CrossEntropyCriterion():float()

local function addTransforms(dataset, mean, std)
   local tnti = vision.TransformImage()
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.path,'/')
      sample.label  = spl[#spl-1]
      sample.target = class2target[sample.label]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         tnti:randomScale{minSize=299,maxSize=330},
         tnti:randomCrop(299),
         tnti:colorNormalize(mean, std)
      }(sample.path)
      return sample
   end)
   return dataset
end

trainset = addTransforms(trainvalset, mean, std)
testset  = addTransforms(testset, mean, std)

torch.save('trainset.t7', trainset)
torch.save('testset.t7', testset)

local function getIterator(mode)
   -- mode = {train,val,test}
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = 3,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
      end,
      closure   = function(threadid)
         local dataset = torch.load(mode..'set.t7')
         return dataset:batch(config.bsize)
      end,
      transform = function(sample)
         sample.target = torch.Tensor(sample.target):view(-1,1)
         return sample
      end
   }
   print('Stats of '..mode..'set')
   for i, v in pairs(iterator:exec('size')) do
      print(i, v)
   end
   return iterator
end

local meter = {
   avgvm = tnt.AverageValueMeter(),
   confm = tnt.ConfusionMeter{k=#classes},
   timem = tnt.TimeMeter{unit = false},
   clerr = tnt.ClassErrorMeter{topk = {1,5}}
}

local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'

local function createLog(mode, pathlog)
   os.execute('mkdir -p '..pathdir)
   local keys = {'loss', 'acc1', 'acc5', 'time'}
   local format = {'%10.5f', '%3.2f', '%3.2f', '%.1f'}
   local log = tnt.Log{
      keys = keys,
      onFlush = {
         logtext{filename=pathlog, keys=keys},
         logtext{keys=keys, format=format},
      },
      onSet = {
         logstatus{filename=pathlog},
         logstatus{}, -- print status to screen
      }
   }
   log:status("Mode "..mode)
   return log
end
local log = {
   train = createLog('train', pathtrainlog),
   test  = createLog('test', pathtestlog)
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
      loss = meter.avgvm:value(),
      acc1 = 100 - meter.clerr:value{k = 1},
      acc5 = 100 - meter.clerr:value{k = 5},
      time = meter.timem:value()
   }
   print(string.format('%s epoch: %i; avg. loss: %2.4f; avg. error: %2.4f',
      engine.mode, engine.epoch, meter.avgvm:value(), meter.clerr:value{k = 1}))
end
engine.hooks.onEnd = function(state)
   print('End of epoch '..engine.epoch..' on '..engine.mode..'set')
   log[engine.mode]:flush()
   print('Confusion Matrix (rows = gt, cols = pred)')
   print(meter.confm:value())
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
local testiter  = getIterator('test')

local bestepoch = {
   clerrtop1 = 100,
   clerrtop5 = 100,
   epochid = 0
}

for epoch = 1, config.nepoch do
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
   print('Testing ...')
   engine.mode = 'test'
   engine:test{
      network   = net,
      iterator  = testiter,
      criterion = criterion,
   }
   if bestepoch.clerrtop1 > meter.clerr:value{k = 1} then
      bestepoch = {
         clerrtop1 = meter.clerr:value{k = 1},
         clerrtop5 = meter.clerr:value{k = 5},
         epochid = epoch,
         confm = meter.confm:value():clone()
      }
      torch.save(pathbestepoch, bestepoch)
      torch.save(pathbestnet, net:clearState())
   end
end
