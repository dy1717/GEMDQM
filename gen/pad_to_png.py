import ROOT as root
import os
import sys 
root.gROOT.SetBatch(root.kTRUE)


for run in range(0,100):
    for mean in [8010,8050,8100,10300,7100,9010,9020,9050,9070,9100,9200,1710,1711,1712,1713,1714]:
    #for mean in [5,50,100,200,500,700,10300,20300,1717,1718,3010,4015]:
    #for mean in [1710,1711,1712,1713,1714,1715,1716,1717]:
        run = str(run)
        mean = str(mean)
        for _, _, filenames in os.walk('Samples'):
            filenames = [f for f in filenames if ('meanHit_' + mean) in f]
            if len(filenames) == 0:
                raise Exception
            _file = 'Samples/' + filenames[0]
        
            histogram = ('ch 1 run %s' % (run))
            fileName = ("image/%s_%s.png" % (mean,run))
            f = root.TFile(_file)
            h = f.Get(histogram)
            h.SetTitle("Occupancy (slice test) ")
            h.GetXaxis().SetTitle("strip")
            h.GetYaxis().SetTitle("ieta")
            c = root.TCanvas()
            h.Draw("colz")
            h.SetStats(0)
            img = root.TImage.Create()
            img.FromPad(c)
            img.WriteImage(fileName)
