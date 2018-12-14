import ROOT as root
import os
import sys
import json

def get_json_from_root(mean):

    data = []
    mean = str(mean)
    for _, _, filenames in os.walk('Samples'):
        filenames = [f for f in filenames if ('meanHit_' + mean) in f]
        if len(filenames) == 0:
            raise Exception
        _file = 'Samples/' + filenames[0]

    for run in range(0,100):
        histogram = ('ch 1 run %s' % run)
        f = root.TFile(_file)
        h = f.Get(histogram)
        dimx = h.GetNbinsX()
        dimy = h.GetNbinsY()
        contents = []
        for vf in range(1,25):
            if vf % 3 ==1:
                xfrom = 0
                vy = (vf-1)/3
            if vf % 3 ==2:
                xfrom = 32
                vy = (vf-2)/3
            if vf % 3 ==0:
                xfrom = 64
                vy = (vf-3)/3
            vfat = []
            for vx in range(xfrom,xfrom+32):
                vfat.append(h.GetBinContent(vx+1, vy+1))
            contents = vfat
            data.append({ "vfat":str(vf), "disti" : str(mean) , "run": str(run), "content": str(contents)})
    return data
for MEAN in [7100,9010,9020,9050,9070,9100,9200,8010,8050,8100,10300,1710,1711,1712,1713,1714]:
    MEAN = str(MEAN)
    fileName = ("data/%s.json" % (MEAN))
    with open (fileName, 'w+') as file_:
        file_.write(json.dumps(get_json_from_root(MEAN)))
