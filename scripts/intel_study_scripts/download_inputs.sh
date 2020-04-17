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

#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-urand.sgr ${INPUT_DIR}/

#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-kron.sgr ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-kron.sgr.triangles ${INPUT_DIR}/

#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.csgr.triangles ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.gr ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.sgr ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-twitter.tgr ${INPUT_DIR}/

#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.csgr.triangles ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.gr ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.sgr ${INPUT_DIR}/
#wget https://intel-study-sc20-galois-inputs.s3.us-east-2.amazonaws.com/GAP-web.tgr ${INPUT_DIR}/




