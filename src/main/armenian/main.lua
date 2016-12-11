require 'os'
require 'image'
require 'optim'
ffi = require 'ffi'
local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
local transformimage = require 'torchnet-vision.image.transformimage'

require 'src.models.simplecnn'

local cmd = torch.CmdLine()
cmd:option('-cpu', false, 'use cpu for training')
cmd:option('-seed', 1337, 'seed')
cmd:option('-lr', 3e-4, 'learning rate for adam')
cmd:option('-nepoch', 50, 'number of epochs')
cmd:option('-bsize', 50, 'number of images per batch')
cmd:option('-nthread', 5, 'number of threads loading batches')
local config = cmd:parse(arg)
print(string.format('running on %s', config.cpu and 'CPU' or 'GPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.date  = os.date("%y_%m_%d_%X")
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local pathdata = paths.concat('data','armenian')
local pathlog = paths.concat('logs','armenian',config.date)
local pathtrainlog  = paths.concat(pathlog,'trainlog.txt')
local pathtestlog   = paths.concat(pathlog,'testlog.txt')
local pathbestepoch = paths.concat(pathlog,'bestepoch.t7')
local pathbestnet   = paths.concat(pathlog,'net.t7')
local pathconfig    = paths.concat(pathlog,'config.t7')
os.execute('mkdir -p '..pathlog)
torch.save(pathconfig, config)

--------------------------------------
-- Dataset
--------------------------------------

local armenian = require 'src.data.armenian'
local trainset, testset, classes, class2target = armenian.load(pathdata)

local function addTransforms(dataset, mean, std)
   dataset = dataset:transform(function(sample)
      sample.input  = tnt.transform.compose{
         function(path)
            local img = image.load(path)
            if not (img:size(1) == 1) then
               local new_img = torch.zeros(1,img:size(2),img:size(3))
               for i=1, img:size(1) do
                  new_img[1] = new_img[1] + img[i]
               end
               new_img[1] = new_img[1] / img:size(1)
               img = new_img
            end
            return img
         end,
         transformimage.randomScale{
            minSize = 100,
            maxSize = 120
         },
         transformimage.randomCrop(100)
         --transformimage.colorNormalize(mean, std)
      }(sample.path)
      return sample
   end)
   return dataset
end

-- TODO add normalization mean and std
-- local mean, std
-- add mean and std pointers to nil values
-- so colorNormalize return identity
trainset = trainset:shuffle()
trainset = addTransforms(trainset, mean, std)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
testset  = addTransforms(testset, mean, std)

local function getIterator(dataset)
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = config.nthread,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
         torch.setdefaulttensortype('torch.FloatTensor')
      end,
      closure   = function(threadid)
         return dataset:batch(config.bsize)
      end,
      transform = function(sample)
         sample.target = torch.Tensor(sample.target):view(-1,1)
         return sample
      end
   }
   print('Stats')
   for i, v in pairs(iterator:exec('size')) do
      print('Theadid='..i, 'Batch number='..v)
   end
   return iterator
end

--------------------------------------
-- Model
--------------------------------------

local net = simpleCnn(#classes):float()
print(net)

local criterion = nn.CrossEntropyCriterion():float()

--------------------------------------
-- Meter and log
--------------------------------------

local meter = {
   avgvm = tnt.AverageValueMeter(),
   confm = tnt.ConfusionMeter{k=#classes, normalized=true},
   timem = tnt.TimeMeter{unit = false},
   clerr = tnt.ClassErrorMeter{topk = {1,5}}
}

local function createLog(mode, pathlog)
   local keys = {'epoch', 'loss', 'acc1', 'acc5', 'time'}
   local format = {'%d', '%10.5f', '%3.2f', '%3.2f', '%.1f'}
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

--------------------------------------
-- Engine
--------------------------------------

local engine = tnt.OptimEngine()
engine.hooks.onStart = function(state)
   for _, m in pairs(meter) do m:reset() end
end
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
   print(string.format('%s epoch: %i _ avg. loss: %2.4f _ avg. acctop1: %2.4f%%',
      engine.mode, engine.epoch, meter.avgvm:value(), 100 - meter.clerr:value{k = 1}))
end
engine.hooks.onEnd = function(state)
   print('End of epoch '..engine.epoch..' on '..engine.mode..'set')
   log[engine.mode]:flush()
   print('Confusion matrix saved (rows = gt, cols = pred)\n')
   image.save(pathlog..'/confm_epoch,'..engine.epoch..'.pgm', meter.confm:value())
end
if not config.cpu then
   require 'cutorch'
   cutorch.manualSeed(config.seed)
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
   end
end

local bestepoch = {
   acctop1 = 0,
   acctop5 = 0,
   epoch = 0
}

local trainiter = getIterator(trainset)
local testiter  = getIterator(testset)

for epoch = 1, config.nepoch do
   print('Training ...')
   engine.mode = 'train'
   trainiter:exec('manualSeed', config.seed + epoch) -- call trainset:manualSeed(seed)
   trainiter:exec('resample')
   engine:train{
      network     = net,
      iterator    = trainiter,
      criterion   = criterion,
      optimMethod = optim.adam,
      config      = {
         learningRate = config.lr,
         learningRateDecay = 0
      },
      maxepoch    = 1,
   }
   print('Testing testset ...')
   engine.mode = 'test'
   engine:test{
      network   = net,
      iterator  = testiter,
      criterion = criterion,
   }
   if bestepoch.acctop1 > meter.clerr:value{k = 1} then
      bestepoch = {
         acctop1 = 100 - meter.clerr:value{k = 1},
         acctop5 = 100 - meter.clerr:value{k = 5},
         epoch = epoch,
         confm = meter.confm:value():clone()
      }
      torch.save(pathbestepoch, bestepoch)
      torch.save(pathbestnet, net:clearState())
   end
end
