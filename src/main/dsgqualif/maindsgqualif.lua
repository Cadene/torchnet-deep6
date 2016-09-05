local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
require 'image'
require 'os'
require 'optim'
ffi = require 'ffi'
unistd = require 'posix.unistd'
local lsplit    = string.split
local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
local utils     = require 'src.data.utils'
local dsgqualif = require 'src.data.dsgqualif'

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 17, 'batch size')
cmd:option('-nepoch', 50, 'epoch number')
cmd:option('-lr', 1e-3, 'learning rate for adam')
cmd:option('-lrd', 0, 'learning rate decay')
cmd:option('-ftfactor', 1, 'fine tuning factor')
cmd:option('-fromscratch', true, 'reset net')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unistd.getpid()
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '/net/big/cadene/doc/Deep6Framework2'
local pathmodel = path..'/models/raw/inceptionv3/net.t7'
local pathdataset = path..'/data/processed/dsgqualif'
local pathtrainset = pathdataset..'/trainset.t7'
local pathtestset  = pathdataset..'/testset.t7'
local pathvalset   = pathdataset..'/valset.t7'
os.execute('mkdir -p '..pathdataset)

local pathlog = path..'/logs/dsgqualif/'..config.date
local pathtrainlog  = pathlog..'/trainlog.txt'
local pathvallog    = pathlog..'/vallog.txt'
local pathtestlog   = pathlog..'/testlog.txt'
local pathbestepoch = pathlog..'/bestepoch.t7'
local pathbestnet   = pathlog..'/net.t7'
local pathconfig    = pathlog..'/config.t7'
os.execute('mkdir -p '..pathlog)
torch.save(pathconfig, config)

local trainset, valset, testset, classes, class2target = dsgqualif.load()

local net = vision.models.inceptionv3.loadFinetuning{
   filename = pathmodel,
   ftfactor = config.ftfactor,
   nclasses = #classes
}
local criterion = nn.CrossEntropyCriterion():float()
if config.fromscratch then
   print('Reset network', net:reset())
end
print(net)

local function addTransforms(dataset)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.path,'/')
      sample.label  = spl[#spl-1]
      sample.target = class2target[sample.label]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.randomScale{minSize=299,maxSize=310},
         vision.image.transformimage.randomCrop(299),
         vision.image.transformimage.horizontalFlip(),
         vision.image.transformimage.verticalFlip(),
         vision.image.transformimage.rotation(0.1),
         vision.image.transformimage.colorNormalize{
            mean = vision.models.inceptionv3.mean,
            std  = vision.models.inceptionv3.std
         }
      }(sample.path)
      return sample
   end)
   return dataset
end

trainset = trainset:shuffle()--(300)
-- valset   = valset:shuffle(300)
-- testset  = testset:shuffle(300)
trainset = addTransforms(trainset)
valset   = addTransforms(valset)
-- testset  = addTransforms(testset)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
torch.save(pathtrainset, trainset)
torch.save(pathvalset, valset)
-- torch.save(pathtestset, testset)

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
   clerr = tnt.ClassErrorMeter{topk = {1}}
}

local function createLog(mode, pathlog)
   local keys = {'epoch', 'loss', 'acc1', 'time'}
   local format = {'%d', '%10.5f', '%3.2f', '%.1f'}
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
   val   = createLog('val', pathvallog),
   -- test  = createLog('test', pathtestlog)
}

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
      epoch = engine.epoch,
      loss  = meter.avgvm:value(),
      acc1  = 100 - meter.clerr:value{k = 1},
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
local valiter   = getIterator('val')
-- local testiter  = getIterator('test')

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
   print('Validating ...')
   engine.mode = 'val'
   engine:test{
      network   = net,
      iterator  = valiter,
      criterion = criterion,
   }
   if bestepoch.clerrtop1 > meter.clerr:value{k = 1} then
      bestepoch = {
         clerrtop1 = meter.clerr:value{k = 1},
         epoch = epoch,
         confm = meter.confm:value():clone()
      }
      torch.save(pathbestepoch, bestepoch)
      torch.save(pathbestnet, net:clearState())
   end
end
