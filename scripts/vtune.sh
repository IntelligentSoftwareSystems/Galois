#!/bin/bash

# README
# run as:
# threads="1 2 4 8 16" vtune.sh program ARGS

TSTEP="5";
tmax="40";
threads=${threads:="1 `seq $TSTEP $TSTEP $TMAX`"};

prefix=$(basename $1); # assuming arg 1 is the path to the program being run

scriptsDir=$(dirname $0)

echo "scriptsDir=$scriptsDir"

stamp=$(date +'%Y-%m-%d_%H:%M:%S')

OUT_PREFIX="${prefix}_vtune_out_${stamp}"

for t in $threads; do
  # outfile="${OUT_PREFIX}${t}"

  cmd="$scriptsDir/run_vtune.pl -t $t -- ${OUT_PREFIX} $@"

  date;
  echo "Running: $cmd"
  $cmd
done 2>&1 | tee ${OUT_PREFIX}.run.log

function_out="${OUT_PREFIX}.function.log";
line_out="${OUT_PREFIX}.line.log";

SUMM_PREFIX="${prefix}_vtune_summary";

cat $function_out | c++filt | perl $scriptsDir/report_vtune.pl --in function > ${SUMM_PREFIX}.function.${stamp}.csv

cat $line_out | perl $scriptsDir/report_vtune.pl --in line > ${SUMM_PREFIX}.line.${stamp}.csv


# for t in $threads; do
  # function_out="${OUT_PREFIX}${t}.function.log"
  # cat $function_out;
# done | perl $scriptsDir/report_vtune.pl --in function > vtune_summary.function.${stamp}.csv

# for t in $threads; do
  # line_out="${OUT_PREFIX}${t}.line.log"
  # cat $line_out;
# done | perl $scriptsDir/report_vtune.pl --in line > vtune_summary.line.${stamp}.csv
  


