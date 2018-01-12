#!/bin/bash

for i in {1..3}; do python ../bmk2/test2.py --max-output-bytes 0 --log ${BMK_LOGS}/bmkrunlog${i}.log --verbose run; done 
