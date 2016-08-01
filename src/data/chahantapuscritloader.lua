function dataInterim(pathdata, pathdataraw)
   os.execute('mkdir -p '..pathdata..';'
            ..'unzip '..pathdataraw..'/BDD_Remi_Tapuscrit.zip'
               ..' -d '..pathdata)
   -- mv train and test
end
