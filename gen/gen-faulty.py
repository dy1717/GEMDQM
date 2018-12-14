import ROOT as root
import random

### make fake faulty data similar to real data ###
for mean in [10,50,100]:
    totalRuns = 100
    
    h2f = []
    for i in range(0,totalRuns):
        h2f.append(root.TH2F())
    
    sigma = mean/3
    
    f = root.TFile("Samples/genOcc_meanHit_%i.root" % (mean+8000),"NEW")
    eventGenerator = root.TRandom()
    
    for run in range(0,totalRuns):
        h2f[run]  = root.TH2F("ch %i run %i "%(1, run),"occupancy",96,0.5,96.5,8,0.5,8.5)
        h2f[run].GetXaxis().SetTitle("strip")
        h2f[run].GetYaxis().SetTitle("ieta")
        for roll in range(1,9):
            for strip in range(1,97):
                hits = int(eventGenerator.Gaus(mean*(1+roll/40),sigma))
                if hits < 0 : hits = 0
                h2f[run].SetBinContent(strip,roll,hits)
        nFaults = random.randrange(1,6)
        faultyVfats = random.sample(list(range(1,25)), nFaults)
        for fv in faultyVfats:
            if fv % 3 == 1:
                xfrom = 0 
                vy = (fv-1)/3
            if fv % 3 == 2:
                xfrom = 32
                vy = (fv-2)/3
            if fv % 3 == 0:
                xfrom = 64 
                vy = (fv-3)/3
            for vx in range(xfrom,xfrom+32):
                h2f[run].SetBinContent(vx+1, vy+1, 0)
    
        num_zero = random.randrange(350,400)
        for i in range(num_zero):
    
            rx = random.randrange(1,97)
            ry = random.randrange(1,9)
            h2f[run].SetBinContent(rx,ry,0)
    
        foo = ''
        if run < 30:
            for vfat in range(1,25):
                if vfat in faultyVfats:
                    score = 0 
                else:
                    score = 1 
                csv = "%s,%s,%s,%s" % (run,mean+8000,vfat,score)
                foo += "\n" +csv
        with open("labels-sim.csv", "ab") as file_:
            file_.write(foo)
    
    f.Write()
    f.Close()
    
