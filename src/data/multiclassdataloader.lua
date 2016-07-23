require 'torch'
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
require 'sys'
require 'xlua'
require 'image'

local DataLoaderMultiClass = torch.class('DataLoaderMultiClass', 'DataLoader')

function DataLoaderMultiClass:__init()
   DataLoader.__init(self)
end

-- forceClasses can be nil
function DataLoaderMultiClass:create_dataset(pathRoot, type, forceClasses)
   local name = type .. 'set'
   local pathType = pathRoot..'/'..type
   local classes, classLabels = self:load_classes(pathType, forceClasses)
   local pathImages, labelImages = self:find_data(pathType, classes, classLabels)
   return {
      classes = classes,
      classLabels = classLabels,
      pathImages = pathImages,
      labelImages = labelImages,
      name = name,
      size = pathImages:size(1)
   }
end

local function table_find(t, o)
   for k,v in pairs(t) do
      if v == o then
         return k
      end
   end
end

function DataLoaderMultiClass:find_classes(pathRoot)
   local classes = {}
   local dirs = dir.getdirectories(pathRoot)
   for k,dirpath in ipairs(dirs) do
      local class = paths.basename(dirpath)
      local idx = table_find(classes, class)
      if not idx then
         table.insert(classes, class)
      end
   end
   return classes
end

function DataLoader:is_class_dir(pathRoot)
   local classes = {}
   local dirs = dir.getdirectories(pathRoot)
   for k,dirpath in ipairs(dirs) do
      local class = paths.basename(dirpath)
      local idx = table_find(classes, class)
      if not idx then
         table.insert(classes, class)
      end
   end
   return #classes ~= 0 -- true if classes
end


function DataLoaderMultiClass:find_data(pathRoot, classes, classLabels)
   local classes = classes
   if not classLabels then
      classes = {'nil'}
   end
   local pathImages = torch.CharTensor()
   local labelImages = torch.LongTensor()

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if ffi.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   -- options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i = 2, #extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- print('running "find" on each class directory, and concatenate all'
   --       .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i = 1, #classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();

   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(classes) do
      -- iterate over self.classPaths
      local pathClass
      if class == 'nil' then
         pathClass = pathRoot
      else
         pathClass = pathRoot..'/'..class
      end
      local command = find .. ' "' .. pathClass .. '" ' .. findOptions
         .. ' >>"' .. classFindFiles[i] .. '" \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   -- print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i = 1, #classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   -- print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   pathImages:resize(length, maxPathLength):fill(0)
   local s_data = pathImages:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end

   local nData = pathImages:size(1)
   if self.verbose then print(nData ..  ' samples found.') end

   -- print('Updating classList and imageClass appropriately')
   labelImages:resize(nData)
   local runningIndex = 0
   for i = 1, #classes do
      if self.verbose then xlua.progress(i, #classes) end
      local length = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         -- self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         labelImages[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   -- print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i = 1, #classFindFiles do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')

   return pathImages, labelImages
end

return DataLoaderMultiClass
