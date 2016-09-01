local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
require 'image'
require 'os'
require 'optim'
require 'src.utils'
ffi = require 'ffi'
unistd = require 'posix.unistd'
local lsplit    = string.split
local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
local utils     = require 'torchnet-vision.datasets.utils'

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 25, 'batch size')
cmd:option('-nepoch', 50, 'epoch number')
cmd:option('-lr', 0.0003, 'learning rate for adam')
cmd:option('-lrd', 0, 'learning rate decay')
cmd:option('-ftfactor', 10, 'fine tuning factor')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unistd.getpid()
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '.'
local pathdata     = '../../public/upmcfood101_224'
local pathvggm     = path..'/models/raw/vggm/net.t7'
local pathdataset  = path..'/data/processed/upmcfood101'
local pathtrainset = pathdataset..'/trainset.t7'
local pathtestset  = pathdataset..'/testset.t7'
os.execute('mkdir -p '..pathdataset)

local pathlog = path..'/logs/upmcfood101marion/'..config.date
local pathtrainlog  = pathlog..'/trainlog.txt'
local pathtestlog   = pathlog..'/testlog.txt'
local pathbestepoch = pathlog..'/bestepoch.t7'
local pathbestnet   = pathlog..'/net.t7'
local pathconfig    = pathlog..'/config.t7'
os.execute('mkdir -p '..pathlog)
torch.save(pathconfig, config)

local trainset, classes, class2target = utils.loadDataset(pathdata..'/train')
local testset, _, _                   = utils.loadDataset(pathdata..'/test')

local net = vision.models.vggm.loadFinetuning{
   filename = pathvggm,
   ftfactor = config.ftfactor,
   nclasses = #classes
}
print(net)
local mean = vision.models.inceptionv3.mean
local std  = vision.models.inceptionv3.std
local criterion = nn.CrossEntropyCriterion():float()

local function addTransforms(dataset, mean, std)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.path,'/')
      sample.label  = spl[#spl-1]
      sample.target = class2target[sample.label]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.randomScale{minSize=221,maxSize=230},
         vision.image.transformimage.randomCrop(221),
         vision.image.transformimage.moveColor(),
         function(img) return img * 255 end,
         vision.image.transformimage.colorNormalize(mean, std)
      }(sample.path)
      return sample
   end)
   return dataset
end

trainset = trainset:shuffle()--(300)
trainset = addTransforms(trainset, mean, std)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
-- testset  = testset:shuffle(300)
testset  = addTransforms(testset, mean, std)

torch.save(pathtrainset, trainset)
torch.save(pathtestset, testset)

local function getIterator(mode)
   -- mode = {train,val,test}
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = config.nthread,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
      end,
      closure   = function(threadid)
         local dataset = torch.load(pathdataset..'/'..mode..'set.t7')
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
      epoch = engine.epoch,
      loss  = meter.avgvm:value(),
      acc1  = 100 - meter.clerr:value{k = 1},
      acc5  = 100 - meter.clerr:value{k = 5},
      time  = meter.timem:value()
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
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- Iterator
local trainiter = getIterator('train')
local testiter  = getIterator('test')

local bestepoch = {
   clerrtop1 = 100,
   clerrtop5 = 100,
   epoch = 0
}

for epoch = 1, config.nepoch do
   print('Training ...')
   engine.mode = 'train'
   trainiter:exec('manualSeed', config.seed + epoch)
   trainiter:exec('resample')
   engine:train{
      maxepoch    = 1,
      network     = net,
      iterator    = trainiter,
      criterion   = criterion,
      optimMethod = optim.adam,
      config      = {
         learningRate      = config.lr,
         learningRateDecay = config.lrd
      },
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
         epoch = epoch,
         confm = meter.confm:value():clone()
      }
      torch.save(pathbestepoch, bestepoch)
      torch.save(pathbestnet, net:clearState())
   end
end
