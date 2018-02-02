#!/bin/bash

for i in {0..234..1};
do
    python batchCSV.py $i
done

python CSVs/cat.py
