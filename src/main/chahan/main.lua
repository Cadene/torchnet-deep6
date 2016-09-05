local tnt = require 'torchnet'
local os = require 'os'
local optim = require 'optim'
local ffi = require 'ffi'
local image = require 'image'

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', true, 'use gpu for training')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

local env = {}
env.root = '/net/big/cadene/doc/Deep6Framework2'
env.data = env.root .. '/data'
env.dataraw = env.data .. '/raw'
env.datainterim = env.data .. '/interim'
env.dataprocessed = env.data .. '/processed'
env.dataname = 'Chahan'

local function dataInterim()
   os.execute('mkdir -p '..env.datainterim..'/'..env.dataname..';'
            ..'unzip '..env.dataraw..'/'..env.dataname..'/BDD_Remi_Tapuscrit.zip'
               ..' -d '..env.datainterim..'/'..env.dataname)
   -- mv train and test
end

-- require 'src.data.dataloader'
-- require 'src.data.multiclassdataloader'

-- local function dataProcess()
--    local loader = DataLoaderMultiClass()
--    local traindataset = loader:create_dataset(env.datainterim..'/'..env.dataname, 'train')
--    local testdataset = loader:create_dataset(env.datainterim..'/'..env.dataname, 'test', traindataset.classes)
--
--    local function writeDataset(dataset)
--       local dirname = dataset.name..'dataset'
--       local pathdata = env.dataprocessed..'/'..dirname..'/'..dirname
--       os.execute('mkdir -p '..pathdata)
--       local inputwriter = tnt.IndexedDatasetWriter{
--          indexfilename = pathdata..'/input.idx',
--          datafilename  = pathdata..'/input.bin',
--          type          = 'float'
--       }
--       local targetwriter = tnt.IndexedDatasetWriter{
--          indexfilename = pathdata..'/target.idx',
--          datafilename  = pathdata..'/target.bin',
--          type          = 'byte'
--       }
--       for i = 1, testdataset.size do
--          local pathimg = ffi.string(torch.data(dataset.pathImages[i]))
--          inputwriter:add(image.load(pathImg))
--          targetwriter:add(dataset.labelImages[i])
--       end
--       inputwriter:close()
--       targetwriter:close()
--    end
--
--    writeDataset(traindataset)
--    writeDataset(testdataset)
-- end

--dataProcess()

-- Find classes

function findClasses(path)
   local find = 'find'
   if ffi.os == 'OSX' then
      find = 'gfind'
   end
   local handle = io.popen(find..' '..path..' -mindepth 1 -maxdepth 1 -type d'
                           ..' | cut -d/ -f11 | sort')
   local classes = {}
   local class2target = {}
   local key = 1
   for class in handle:lines() do
      table.insert(classes, class)
      class2target[class] = key
      key = key + 1
   end
   handle:close()
   return classes, class2target
end

function findFilenames(path, classes, filename)
   local filename = filename or 'filename.txt'
   local pathfilename = path..'/'..filename
   local find = 'find'
   if ffi.os == 'OSX' then
      find = 'gfind'
   end
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i = 2, #extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end
   os.execute('rm '..pathfilename)
   for _, class in pairs(classes) do
      os.execute(find..' "'..path..'/'..class..'" '..findOptions
                 ..' | cut -f11-12 -d/ >> '..pathfilename)
   end
   return pathfilename
end


function saveDataset(path, mode)
   local modesub = mode == 'trainval' and 'train' or 'test'
   local path = path .. '/' .. modesub
   local classes, class2target = findClasses(path)
   local pathfilename = findFilenames(path, classes, 'filename.txt')
   local save = {
      classes      = classes,
      class2target = class2target,
      pathfilename = pathfilename,
      path         = path
   }
   torch.save(mode..'.t7', save)
   return classes
end

local path = '/net/big/cadene/doc/Deep6Framework2/data/interim/Chahan'
local classes = saveDataset(path, 'trainval')
                saveDataset(path, 'test')

local function getIterator(modesub)
   -- modesub = {train,val,test}
   return tnt.ParallelDatasetIterator{
      nthread   = 3,
      init      = function()
         require 'torchnet'
      end,
      closure   = function(threadid)
         torch.setdefaulttensortype('torch.FloatTensor')
         local ffi   = require 'ffi'
         local os    = require 'os'
         local image = require 'image'
         local mode = (modesub == 'train' or modesub == 'val') and 'trainval' or 'test'
         local save = torch.load(mode..'.t7')
         local classes      = save.classes
         local class2target = save.class2target
         local pathfilename = save.pathfilename
         local path         = save.path
         local dataset = tnt.ListDataset{
            filename = pathfilename,
            path = path,
            load = function(line)
               local sample = {}
               sample.path = line
               local spl = sample.path:split('/')
               sample.label = spl[#spl-1]
               sample.target = class2target[sample.label]
               sample.input = image.load(sample.path, 3)
               return sample
            end
         }
         dataset = dataset:transform(function(sample)
            sample.input = image.scale(sample.input, 100, 100)
            return sample
         end)
         if mode == 'trainval' then
            torch.manualSeed(1337)
            dataset = dataset:shuffle()
            dataset = dataset:split{
               train = 0.7,
               val   = 0.3
            }
            dataset:select(modesub)
         end
         dataset = dataset:batch(30)
         return dataset
      end,
      transform = function(sample)
         sample.target = torch.Tensor(sample.target):view(-1,1)
         return sample
         -- sample.input = image.scale(sample.input, 20, 20)
      end
   }
end


local net = nn.Sequential()
net:add(nn.SpatialConvolution(3,32,5,5))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialBatchNormalization(32))
net:add(nn.SpatialConvolution(32,64,5,5))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialBatchNormalization(64))
net:add(nn.SpatialConvolution(64,128,5,5))
net:add(nn.ReLU(true))
net:add(nn.SpatialBatchNormalization(128))
   -- net:add(nn.SpatialConvolution(32,84,1,1))
   -- net:add(nn.ReLU(true))
   -- net:add(nn.SpatialBatchNormalization(84))
   -- net:add(nn.SpatialConvolution(84,10,1,1))
   -- net:add(WeldonAggregation(config.weldon.kmax,config.weldonkmin))
   -- net:add(nn.View(#classes))
net:add(nn.View(-1,128*18*18))
net:add(nn.Dropout(0.5))
net:add(nn.Linear(128*18*18,1024))
net:add(nn.ReLU(true))
net:add(nn.BatchNormalization(1024))
net:add(nn.Dropout(0.5))
net:add(nn.Linear(1024,#classes))

net:float()
print(net)
print(net:forward(torch.ones(2,3,100,100):float()):size())

local criterion = nn.CrossEntropyCriterion()
criterion:float()

local engine  = tnt.OptimEngine()
local meter   = tnt.AverageValueMeter()
local confmat = tnt.ConfusionMeter{k=#classes}
local clerr   = tnt.ClassErrorMeter{topk = {1}}

-- Hooks

engine.hooks.onStart = function(state)
   meter:reset()
   clerr:reset()
   state.mode = state.training and "train" or "val"
end

engine.hooks.onStartEpoch = function(state) -- training only
   engine.epoch = engine.epoch and (engine.epoch + 1) or 1
end

engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   print(string.format('%s epoch: %i; avg. loss: %2.4f; avg. error: %2.4f',
      state.mode, engine.epoch, meter:value(), clerr:value{k = 1}))
end

engine.hooks.onEnd = function(state)
   print(string.format('%s loss: %2.4f; error: %2.4f',
      state.mode, meter:value(), clerr:value{k = 1}))
end

-- set up GPU training:
if config.usegpu then
   -- copy model to GPU:
   require 'cunn'
   require 'cudnn'
   cudnn.convert(net, cudnn)
   net       = net:cuda()
   criterion = criterion:cuda()
   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- engine.hooks.onEndEpoch = function(state)
--    state.iterator.dataset:select('val')
-- end

-- engine.hooks.onSample = function(state)
--    print(torch.type(state.sample.input))
--    print(state.sample.input:size())
-- end
-- engine.hooks.onForward = function(state)
--    print(state.network.output:size())
--    print(state.sample.target)
-- end

-- Iterator

local trainiter = getIterator('train')
print('Trainset stats')
for i, v in pairs(trainiter:exec("size")) do
   print(i, v)
end

local valiter   = getIterator('val')
print('Valset stats')
for i, v in pairs(valiter:exec("size")) do
   print(i, v)
end

local testiter   = getIterator('test')
print('Testset stats')
for i, v in pairs(testiter:exec("size")) do
   print(i, v)
end

for epoch = 1, 10 do
   print('Training ...')
   engine:train{
      network     = net,
      iterator    = trainiter,
      criterion   = criterion,
      optimMethod = optim.adam,
      config      = {
         learningRate = 1e-3
      },
      maxepoch    = 1,
   }
   print('Testing valset ...')
   engine:test{
      network   = net,
      iterator  = valiter,
      criterion = criterion,
   }
   torch.save('net.t7',net:clearState())
end

print('Testing testset ...')
engine:test{
   network   = net,
   iterator  = testiter,
   criterion = criterion,
}


-- local dataset = tnt.TransformDataset{
--    dataset = dataset,
--    transform = function(sample)
--       sample.input = image.load(sample.path, 3)
--       sample.target = 1
--       return sample
--    end
-- }
-- --
-- print(dataset:get(1))
