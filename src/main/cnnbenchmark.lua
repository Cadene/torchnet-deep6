require 'sys'
require 'nn'
require 'cunn'
require 'cudnn'
vision = require 'torchnet-vision'

local nets = {}
nets[#nets+1] = vision.models.overfeat.load('models/overfeat/net.t7')
nets[#nets+1] = vision.models.vgg16.load('models/vgg16/net.t7')
nets[#nets+1] = vision.models.inceptionv3.load('models/inceptionv3/net.t7')

local nets_name = {}
nets_name[#nets_name+1] = 'Overfeat'
nets_name[#nets_name+1] = 'Vgg16'
nets_name[#nets_name+1] = 'InceptionV3'

local nets_size = {}
nets_size[#nets_size+1] = 224
nets_size[#nets_size+1] = 221
nets_size[#nets_size+1] = 299

local libs = {}
libs[#libs+1] = nn
-- libs[#libs+1] = nn
-- libs[#libs+1] = cudnn

local libs_name = {}
libs_name[#libs_name+1] = 'nn float'
-- libs_name[#libs_name+1] = 'nn cuda'
-- libs_name[#libs_name+1] = 'cudnn'

local libs_GPU = {}
libs_GPU[#libs_GPU+1] = false
-- libs_GPU[#libs_GPU+1] = true
-- libs_GPU[#libs_GPU+1] = true

steps = 10 -- nb of steps in loop to average perf

function makeInput(layout, size)
   local osize
   if layout == 'BDHW' then
      osize = size
   elseif layout == 'DHWB' then
      osize = {size[2],size[3],size[4],size[1]}
   elseif layout == 'BHWD' then
      osize = {size[1], size[3], size[4], size[2]}
   end
   return torch.randn(torch.LongStorage(osize))
end

for i=1,#nets do
   for j=1,#libs do
      collectgarbage()
      local model = nets[i]:float()
      local model_name = nets_name[i]
      local size = {50, 3, nets_size[i], nets_size[i]}
      local input = makeInput('BDHW',size):float()
      local lib_name = libs_name[j]

      cudnn.convert(model, libs[j])

      if libs_GPU[j] then
         model=model:cuda()
         input=input:cuda()
      end

      print('ModelType: ' .. model_name, 'Implem: ' .. lib_name, 
            'Input shape: ' .. input:size(1) .. 'x' .. input:size(2) .. 
               'x' .. input:size(3) .. 'x' .. input:size(4))
      
      -- dry-run
      model:zeroGradParameters()
      local output = model:updateOutput(input)
      local gradInput = model:updateGradInput(input, output)
      model:accGradParameters(input, output)
      if libs_GPU[j] then cutorch.synchronize() end
      collectgarbage()
      
      local tmf, tmbi, tmbg
      sys.tic()
      for t = 1,steps do
         output = model:updateOutput(input)
      end
      if libs_GPU[j] then cutorch.synchronize() end
      tmf = sys.toc()/steps
      print(string.format("%-30s %25s %10.2f", lib_name, ':updateOutput():', tmf*1000))

      collectgarbage()
      sys.tic()
      for t = 1,steps do
         model:updateGradInput(input, output)
      end
      if libs_GPU[j] then cutorch.synchronize() end
      tmbi = sys.toc()/steps
      print(string.format("%-30s %25s %10.2f", lib_name, ':updateGradInput():', tmbi*1000))

      collectgarbage()
      sys.tic()
      local ok = 1
      for t = 1,steps do
         ok = pcall(function() model:accGradParameters(input, output) end)
      end
      if libs_GPU[j] then cutorch.synchronize() end
      tmbg = sys.toc()/steps
      if not ok then
         print(string.format("%-30s %25s %s", lib_name, ':accGradParameters():', 'FAILED!'))
      else
         print(string.format("%-30s %25s %10.2f", lib_name, ':accGradParameters():', tmbg*1000))
      end
      print(string.format("%-30s %25s %10.2f", lib_name, ':Forward:', (tmf)*1000))
      print(string.format("%-30s %25s %10.2f", lib_name, ':Backward:', (tmbi+tmbg)*1000))
      print(string.format("%-30s %25s %10.2f", lib_name, ':TOTAL:', (tmf+tmbi+tmbg)*1000))
      print()

      model:clearState()
      model:float()
   end
end

print('')

