#!/bin/bash

echo -e "USAGE: ./download_inputs.sh INPUT_DIR_PATH\n"
INPUT_DIR=$1
if [ -z ${INPUT_DIR} ];
then
  echo "INPUT_DIR not set; Please point it to the directory where graphs will be downloaded"
  exit
else
  echo "Using directory ${INPUT_DIR} for inputs"
fi

cd ${INPUT_DIR}
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/sources.tar.gz 
tar -xzvf sources.tar.gz

wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-road.sgr

wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-urand.sgr 

wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-kron.sgr 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-kron.sgr.triangles 

wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.csgr.triangles 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.gr 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.sgr 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.tgr 

wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.csgr.triangles 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.gr 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.sgr 
wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.tgr




