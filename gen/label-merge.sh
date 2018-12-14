#!/bin/bash
# merge fake and real csv file into one file
for i in {sim,real}
do
  cat labels-${i}.csv
done > data/labels.csv
