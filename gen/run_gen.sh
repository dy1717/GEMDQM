#!/bin/bash
cat/dev/null > labels-sim.csv
python gen-good.py
python gen-faulty.py
python gen-allfaulty.py
python gen-hot.py
