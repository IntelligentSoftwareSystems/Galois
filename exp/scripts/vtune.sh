#!/bin/bash

tmin=1;
tinc=1;
tmax=24;

# threads=$(seq $tmin $tinc $tmax)
threads="1 4 8 12 16 20 24 28 32 36 40"
# threads="1 2 4 6 8 10 12 14 16 18 20 22 24"
# threads="1 2 4 8 16 32 40 60 80 100 120 122 140 160 180 200 220 240 244" 
# threads="1 2 4 8"



scriptsDir=$(dirname $0)

echo "scriptsDir=$scriptsDir"

stamp=$(date +'%Y-%m-%d_%H:%M:%S')

OUT_PREFIX="vtune_out_${stamp}"

for t in $threads; do
  # outfile="${OUT_PREFIX}${t}"

  cmd="$scriptsDir/run_vtune.pl -t $t ${OUT_PREFIX} $@"

  date;
  echo "Running: $cmd"
  $cmd
done 2>&1 | tee ${OUT_PREFIX}.run.log

function_out="${OUT_PREFIX}.function.log";
line_out="${OUT_PREFIX}.line.log";

cat $function_out | perl $scriptsDir/report_vtune.pl --in function > vtune_summary.function.${stamp}.csv

cat $line_out | perl $scriptsDir/report_vtune.pl --in line > vtune_summary.line.${stamp}.csv


# for t in $threads; do
  # function_out="${OUT_PREFIX}${t}.function.log"
  # cat $function_out;
# done | perl $scriptsDir/report_vtune.pl --in function > vtune_summary.function.${stamp}.csv

# for t in $threads; do
  # line_out="${OUT_PREFIX}${t}.line.log"
  # cat $line_out;
# done | perl $scriptsDir/report_vtune.pl --in line > vtune_summary.line.${stamp}.csv
  


