import ROOT as root
import random

### make fake faulty data similar to real data ###
mean = 100
totalRuns = 100

h2f = []
for i in range(0,totalRuns):
    h2f.append(root.TH2F())

sigma = mean/3

f = root.TFile("Samples/genOcc_meanHit_%i.root" % (mean+7000),"NEW")
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
    case = random.randrange(1,3)
    if case == 1:
        faultyVfats = random.sample(list(range(1,25)), 23)
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
    
        num_zero = random.randrange(200,300)
        for i in range(num_zero):
    
            rx = random.randrange(1,97)
            ry = random.randrange(1,9)
            h2f[run].SetBinContent(rx,ry,0)
    if case == 2:
        for zx in range(1,97):
            for zy in range(1,9): 
               h2f[run].SetBinContent(zx,zy,0)
    foo = ''
    if run < 30:
        for vfat in range(1,25):
            score = 0
            csv = "%s,%s,%s,%s" % (run,mean+7000,vfat,score)
            foo += "\n" +csv
    with open("labels-sim.csv", "ab") as file_:
        file_.write(foo)

f.Write()
f.Close()
    
