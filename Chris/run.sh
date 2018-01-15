#!/bin/bash

for i in {0..63..1};
do
    ./read_fits.py $i
done
