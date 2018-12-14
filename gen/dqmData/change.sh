#!/bin/bash

a=1000
for i in *.root; do
  new=$(printf "%i.root" "$a") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let a=a+1
done
