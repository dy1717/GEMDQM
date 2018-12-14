import ROOT as root
import os
import sys
import random

### extract some data from real data ###

for run in range(1000,1100):
    for _, _, filenames in os.walk('goodData2'):
        filenames = [f for f in filenames if (str(run) + '.root') in f]
        if len(filenames) == 0:
            raise Exception
        _file = 'goodData2/' + filenames[0]
    
        for chamber in [27,28,29,30]:
            for layer in [1,2]:
                histogram = ('ch %s layer %s' %
                            (chamber,layer))
        
                f = root.TFile(_file)
                h = f.Get(histogram)
                new_num = 4*(layer-1)+(chamber-27)
                new_run = run-1000
                print new_run
                new_histo = root.TH2F("ch %i run %i" %(1,new_run), "occupancy",96,0.5,96.5,8,0.5,8.5)

                for roll in range(1,9):
                   for four_strip in range(1,97):
                       conts = [] 
                       for m in range(1,5):
                           cont = h.GetBinContent(4*(four_strip-1)+m,roll)
                           conts.append(cont)
                           new_histo.SetBinContent(four_strip,roll,sum(conts))
                   
                g = root.TFile("Samples/genOcc_meanHit_171%s.root" % new_num ,"UPDATE")
                new_histo.Write()
                g.Close()
