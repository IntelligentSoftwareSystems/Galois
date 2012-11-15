#!/bin/bash

tmin=1;
tinc=1;
tmax=24;

# threads=$(seq $tmin $tinc $tmax)
threads="1 2 4 8 12 16 20 24 28 32 36 40"


scriptsDir=$(dirname $0)

echo "scriptsDir=$scriptsDir"

stamp=$(date +'%Y-%m-%d_%H:%M:%S')

OUT_PREFIX="vtune_out_${stamp}_"

# putting in a subprocess might help killing all iterations at once
(for t in $threads; do
  outfile="${OUT_PREFIX}${t}"

  cmd="$scriptsDir/run_vtune.pl -t $t $outfile $@"

  date;
  echo "Running: $cmd"
  $cmd
done;)


for t in $threads; do
  function_out="${OUT_PREFIX}${t}.function.log"
  cat $function_out;
done | perl $scriptsDir/report_vtune.pl --in function > vtune_summary.function.${stamp}.csv


for t in $threads; do
  line_out="${OUT_PREFIX}${t}.line.log"
  cat $line_out;
done | perl $scriptsDir/report_vtune.pl --in line > vtune_summary.line.${stamp}.csv
  


