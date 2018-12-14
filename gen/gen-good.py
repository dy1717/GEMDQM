import ROOT as root
import random

### make fake good data similar to real data ###
for mean in [10,20,50,70,100,200]:
    totalRuns = 100
    
    h2f = []
    for i in range(0,totalRuns):
        h2f.append(root.TH2F())
    
    sigma = mean/3
    f = root.TFile("Samples/genOcc_meanHit_%s.root" % (mean+9000),"NEW")
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
        num_zero = random.randrange(350,400)
        for i in range(num_zero):
    
            rx = random.randrange(1,97)
            ry = random.randrange(1,9)
            h2f[run].SetBinContent(rx,ry,0)
        score = 1
        foo = ''
        if run < 30: 
            for vfat in range(1,25):
                csv = "%s,%s,%s,%s" % (run,mean+9000,vfat,score)
                foo += "\n" +csv
        with open("labels-sim.csv", "ab") as file_:
            file_.write(foo)
     
    f.Write()
    f.Close()

