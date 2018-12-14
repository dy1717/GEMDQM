import ROOT as root
import random

###make data having hot cell & label to them###

totalRuns = 100

h2f = []
for i in range(0,totalRuns):
    h2f.append(root.TH2F())

mean = 300
sigma = mean/3

f = root.TFile("Samples/genOcc_meanHit_10%s.root" % mean,"NEW")
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
    num_zero = random.randrange(300,400)
    for i in range(num_zero):
        rx = random.randrange(1,97)
        ry = random.randrange(1,9)
        h2f[run].SetBinContent(rx,ry,0)

    hx = random.randrange(1,97)
    hy = random.randrange(1,9)
    ori_value = h2f[run].GetBinContent(hx,hy)
    h2f[run].SetBinContent(hx,hy,(ori_value+10)*100)
    foo = ''
    faulty_vfat = 3*(hy-1)+((hx+31)//32)
    if run < 30:
        for vfat in range(1,25):
            if vfat == faulty_vfat:
                score = 0
            else:
                score = 1
            csv = "%s,%s,%s,%s" % (run,mean+10000,vfat,score)
            foo += "\n" +csv
    with open("labels-sim.csv", "ab") as file_:
        file_.write(foo)
    
f.Write()
f.Close()

