local argcheck = require 'argcheck'

local upmcfood101 = {}

upmcfood101.__download = argcheck{
   {name='dirname', type='string'},
   call =
      function(dirname)
         os.execute('mkdir -p '..dirname..'; '
         ..'cp /net/big/cadene/doc/Deep6Framework/data/raw/UPMC_Food101/UPMC_Food101.tar.gz'
           ..' '..dirname..'; '
         ..'tar -xzf '..paths.concat(dirname, 'UPMC_Food101.tar.gz')..' -C '..dirname)
      end
}

upmcfood101.__download('/net/big/cadene/doc/Deep6Framework2/data/raw/upmcfood101')
