local argcheck = require 'argcheck'

local tnt   = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'

local dsgqualif = {}

-- dsgqualif.__download = argcheck{
--    {name='dirname', type='string', default='data/raw/dsgqualif'},
--    call =
--       function(dirname)
--          os.execute('unzip '..dirname..'/roof_images.zip -d '..dirname)
--       end
-- }

dsgqualif.load = argcheck{
   {name='dirname', type='string',
    default='/net/big/cadene/doc/Deep6Framework/data/interim/DSG_qualif'},
   call =
      function(dirname)
         local dirtrain = paths.concat(dirname, 'train90')
         local dirval   = paths.concat(dirname, 'val')
         local dirtest  = paths.concat(dirname, 'test')
         local traintxt = paths.concat(dirname, 'TrainImages.txt')
         local valtxt   = paths.concat(dirname, 'ValImages.txt')
         local testtxt  = paths.concat(dirname, 'TestImages.txt')
         -- sed: 's/search/replace'
         -- \ escape for sed + \ escape for lua = \\
         -- ex: I want to escape / so \\/
         os.execute('ls -A1 '..dirtrain..'/*/*.jpg'..
                    '| sed -r \'s/^.+\\/([0-9])\\//\\1\\//\''..
                    '> '..traintxt)
         os.execute('ls -A1 '..dirval..'/*/*.jpg'..
                    '| sed -r \'s/^.+\\/([0-9])\\//\\1\\//\''..
                    '> '..valtxt)
         os.execute('ls -A1 '..dirtest..'/*.jpg'..
                    '| sed -r \'s/^.+\\///\''..
                    '> '..testtxt)
         local classes, class2target = utils.findClasses(dirval)
         local trainset = tnt.ListDataset{
            filename = traintxt,
            path = dirtrain,
            load = function(line)
               local sample = {path=line}
               return sample
            end
         }
         local valset = tnt.ListDataset{
            filename = valtxt,
            path = dirval,
            load = function(line)
               local sample = {path=line}
               return sample
            end
         }
         local testset = tnt.ListDataset{
            filename = testtxt,
            path = dirtest,
            load = function(line)
               local sample = {path=line}
               return sample
            end
         }
         return trainset, valset, testset, classes, class2target
      end
}

return dsgqualif
