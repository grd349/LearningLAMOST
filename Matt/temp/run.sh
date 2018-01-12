#!/bin/bash

for i in {0..234..1};
do
    python makeCSV.py $i
done

python cat.py
