local argcheck = require 'argcheck'

local tnt   = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'

local m2caiworkflow = {}

-- m2caiworkflow.__download = argcheck{
--    {name='dirname', type='string', default='data/raw/m2caiworkflow'},
--    call =
--       function(dirname)
--          os.execute('unzip '..dirname..'/roof_images.zip -d '..dirname)
--       end
-- }

m2caiworkflow.load = function()
   local pathimages   = '/local/robert/m2cai/workflow/images'
   local pathtraintxt = '/local/robert/m2cai/workflow/dataset2/trainset.txt'
   local pathvaltxt   = '/local/robert/m2cai/workflow/dataset2/valset.txt'
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
   return trainset, valset, classes, class2target
end

return m2caiworkflow
