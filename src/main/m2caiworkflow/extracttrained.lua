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
local transformimage =
   require 'torchnet-vision.image.transformimage'

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 20, 'batch size')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
cmd:option('-pathnet', 'logs/m2caiworkflow/16_08_05_05:03:52/net.t7', '')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unilocal tnt = require 'torchnet'
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '/net/big/cadene/doc/Deep6Framework2'
local pathimages   = '/local/robert/m2cai/workflow/images'
local pathtraintxt = '/local/robert/m2cai/workflow/dataset2/trainset.txt'
local pathvaltxt   = '/local/robert/m2cai/workflow/dataset2/valset.txt'
local pathdataset  = path..'/data/processed/m2caiworkflow'
local pathtrainset = pathdataset..'/trainset.t7'
local pathvalset   = pathdataset..'/valset.t7'
os.execute('mkdir -p '..pathdataset)

local pathlog = path..'/logs/m2caiworkflowextract/'..config.date
local pathconfig = pathlog..'/config.t7'
os.execute('mkdir -p '..pathlog)
torch.save(pathconfig, config)

local trainset = tnt.ListDataset{
    filename = pathtraintxt,
    path = pathimages,
    load = function(line)
       local sample = {line=line}
       return sample
    end
}

local valset = tnt.ListDataset{
    filename = pathvaltxt,
    path = pathimages,
    load = function(line)
       local sample = {line=line}
       return sample
    end
}
local classes = {"TrocarPlacement", "Preparation",
   "CalotTriangleDissection", "ClippingCutting",
   "GallbladderDissection", "GallbladderPackaging",
   "CleaningCoagulation", "GallbladderRetraction"}
local class2target = {}
for k,v in pairs(classes) do class2target[v] = k end

local net
require 'cunn'
require 'cudnn'
net = torch.load(config.pathnet)
print(net)
local criterion = nn.CrossEntropyCriterion():float()

local function addTransforms(dataset)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.line,', ')
      sample.path   = spl[1]
      sample.target = spl[2] + 1
      sample.label  = classes[spl[2] + 1]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.randomScale{minSize=299,maxSize=330},
         vision.image.transformimage.randomCrop(299),
         vision.image.transformimage.colorNormalize{
            mean = vision.models.inceptionv3.mean,
            std  = vision.models.inceptionv3.std
         }
      }(sample.path)
      return sample
   end)
   return dataset
end

-- trainset = trainset:shuffle(30)
trainset = addTransforms(trainset)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
-- valset  = valset:shuffle(30)
valset  = addTransforms(valset)

torch.save(pathtrainset, trainset)
torch.save(pathvalset, valset)

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
   timem = tnt.TimeMeter{unit = false},
}

local engine = tnt.OptimEngine()
local file
engine.hooks.onStart = function(state)
   for _,m in pairs(meter) do m:reset() end
   print(engine.mode)
   file = assert(io.open(pathlog..'/'..engine.mode..'extract.csv', "w"))
   file:write('path;gttarget;gtclass')
   for i=1, #classes do file:write(';pred'..i) end
   file:write("\n")
end
engine.hooks.onForward = function(state)
   local output = state.network.output
   for i=1, output:size(1) do
      file:write(state.sample.path[i]);   file:write(';')
      file:write(state.sample.target[i][1]); file:write(';')
      file:write(state.sample.label[i])
      for j=1, output:size(2) do
         file:write(';'); file:write(output[i][j])
      end
      file:write("\n")
   end
end
engine.hooks.onEnd = function(state)
   print('End of extracting on '..engine.mode..'set')
   print('Took '..meter.timem:value())
   file:close()
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

print('Extracting trainset ...')
engine.mode = 'train'
engine:test{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion
}

print('Extracting testset ...')
engine.mode = 'val'
engine:test{
   network   = net,
   iterator  = getIterator('val'),
   criterion = criterion
}
