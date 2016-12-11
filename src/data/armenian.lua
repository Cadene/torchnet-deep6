local argcheck = require 'argcheck'
local tnt   = require 'torchnet'
local utils  = require 'torchnet-vision.datasets.utils'
local utils2 = require 'src.data.utils'
local lsplit = string.split

local armenian = {}

armenian.__download = argcheck{
   {name='dirname', type='string', default='data/armenian/raw'},
   call =
      function(dirname)
         os.execute('mkdir -p '..dirname)
         os.execute('wget https://www.dropbox.com/s/ldcp1saakfvbjoc/BDD_INALCO_minuscules.zip?dl=1 -P '..dirname)
         os.execute('unzip '..dirname..'/BDD_INALCO_minuscules.zip?dl=1 -d '..dirname)
      end
}

armenian.__convertToPng = argcheck{
   {name='bdd', type='string'},
   {name='interim', type='string'},
   {name='classes', type='table'},
   call =
      function(bdd, interim, classes)
         for i, class in ipairs(classes) do
            local pathbddclass = paths.concat(bdd, class)
            local pathinterimclass = paths.concat(interim, class)
            os.execute('mkdir -p '..pathinterimclass)
            os.execute('for f in '..pathbddclass..'/*.tif; '
               .. 'do convert "$f" '..pathinterimclass..'/"$(basename "$f" .tif).png"; '
               .. 'done')
            os.execute('for f in '..pathbddclass..'/*.tiff; '
               .. 'do convert "$f" '..pathinterimclass..'/"$(basename "$f" .tiff).png"; '
               .. 'done')
         end
      end
}

local function splitDataset(dataset, classes, percent, fnametrain, fnametest)
   local counts = {}
   for i=1, #classes do
      counts[i] = 0
   end 
   utils2.iterOverDataset(dataset, dataset:size(), {
      onSample = function(sample)
         counts[sample.target] = counts[sample.target] + 1
      end
   })

   local percents = {}
   for i=1, #classes do
      percents[i] = math.floor(counts[i] * percent * 1.0 / 100)
   end 

   local ftrain = assert(io.open(fnametrain, 'w'))
   local ftest  = assert(io.open(fnametest, 'w'))

   local counts = {}
   for i=1, #classes do
      counts[i] = 0
   end 
   utils2.iterOverDataset(dataset, dataset:size(), {
      onSample = function(sample)
         counts[sample.target] = counts[sample.target] + 1
         if counts[sample.target] < percents[sample.target] then
            ftrain:write(sample.line)
            ftrain:write("\n")
         else
            ftest:write(sample.line)
            ftest:write("\n")
         end
      end
   })
   ftrain:close()
   ftest:close()
end

armenian.load = argcheck{
   {name='dirname', type='string', default='data/armenian'},
   call =
      function(dirname)
         local raw = paths.concat(dirname, 'raw')
         local bdd = paths.concat(raw, 'BDD_INALCO_minuscules')
         local interim = paths.concat(dirname, 'interim')
         local pathfilename = paths.concat(interim, 'filename.txt')
         local pathfnametrain = paths.concat(interim, 'fnametrain.txt')
         local pathfnametest  = paths.concat(interim, 'fnametest.txt')
         
         if not paths.dirp(dirname) then
            armenian.__download(raw)
         end

         local classes = utils.findClasses(bdd)
         local class2target = {}
         for i, class in ipairs(classes) do
            class2target[class] = i
         end

         if not paths.dirp(interim) then
            armenian.__convertToPng(bdd, interim, classes)
         end
         
         if not paths.filep(pathfilename) then
            utils.findFilenames(interim, classes)
         end

         local load = function(line)
            local splt = lsplit(line, '/')
            local sample = {
               line = line,
               path = paths.concat(interim, line),
               class = splt[1],
               target = class2target[splt[1]]
            }
            return sample
         end

         if not paths.filep(pathfnametrain) and
            not paths.filep(pathfnametest) then

            local dataset = tnt.ListDataset{
               filename = pathfilename,
               load = load
            }
            dataset = dataset:shuffle()

            splitDataset(dataset, classes, 70,
               pathfnametrain, pathfnametest)
         end

         local trainset = tnt.ListDataset{
            filename = pathfnametrain,
            load = load
         }
         local testset = tnt.ListDataset{
            filename = pathfnametest,
            load = load
         }
         return trainset, testset, classes, class2target
      end
}

return armenian
