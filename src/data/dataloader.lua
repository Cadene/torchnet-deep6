require 'torch'
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
require 'sys'
require 'xlua'
require 'image'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init()
end

function DataLoader:load_classes(pathRoot, forceClasses)
   local classes
   if forceClasses then
      classes = {}
      for k,v in pairs(forceClasses) do
         classes[k] = v
      end
   else
      classes = self:find_classes(pathRoot)
   end
   local classLabels
   if self:is_class_dir(pathRoot) then
      classLabels = self:create_classLabels(classes)
   end
   return classes, classLabels
end

function DataLoader:create_classLabels(classes)
   local classLabels = {}
   for k,v in ipairs(classes) do
      classLabels[v] = k
   end
   return classLabels
end

function DataLoader:create_dataset(pathRoot, type, forceClasses)
end

function DataLoader:find_classes(pathRoot)
end

function DataLoader:find_data(pathRoot, classes)
end

return DataLoader
