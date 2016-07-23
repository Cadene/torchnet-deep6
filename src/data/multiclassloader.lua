local tnt = require 'torchnet'

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

function loadDataset(path, mode)
   local path = path .. '/' .. mode
   local classes, class2target = findClasses(path)
   local pathfilename = findFilenames(path, classes, 'filename.txt')
   local dataset = tnt.ListDataset{
      filename = pathfilename,
      path = path,
      load = function(line)
         local sample = {path=line}
         return sample
      end
   }
   return dataset, classes, class2target
end
