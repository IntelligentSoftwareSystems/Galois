#!/bin/bash
# Wrapper script around calibrator or oprofile, used to calculate the number
# of cycles it takes to handle a tlb miss.  calibrator will need to be
# downloaded seperately to be used here, otherwise oprofile will be used.
# oprofile does not generate accurate results on x86 or x86_64.
#
# Both methods were lifted from a paper by Mel Gorman <mel@csn.ul.ie>
#
# Licensed under LGPL 2.1 as packaged with libhugetlbfs
# (c) Eric B Munson 2009
# (c) Mel Gorman 2009

# calibrator can be found here:
# http://homepages.cwi.nl/~manegold/Calibrator/v0.9e/calibrator.c
# and should be compiled with this command line:
# gcc calibrator.c -lm -o calibrator
# and then placed in the same directory as this script
# Note: Do not use any optimisation to avoid skewing the results

# trace == 3
# info  == 2 (default, should remain quiet in practicet)
# error == 1
VERBOSE=2
MHZ=0

cpumhz() {
	MAX_MHZ=0
	SYSFS_SCALING=/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

	# Use sysfs if available
	if [ -e $SYSFS_SCALING ]; then
		for CURR_MHZ in `cat $SYSFS_SCALING`; do
			CURR_MHZ=$(($CURR_MHZ/1000))
			if [ $CURR_MHZ -gt $MAX_MHZ ]; then
				MAX_MHZ=$CURR_MHZ
			fi
		done
		MHZ=$MAX_MHZ
		return
	fi

	# Otherwise, use /proc/cpuinfo. Guess what field name is needed.
	# In most cases, it's cpu MHz but there will be exceptions
	FNAME="cpu MHz"
	FINDEX=4
	case "`uname -m`" in
		ppc64)
			FNAME="clock"
			FINDEX=3
			;;
	esac

	# Take a hundred samples in case of CPU frequency scaling artifically
	# returning a low value. The multiple samples should wake up the CPU
	for SAMPLE in `seq 1 100`; do
		for CURR_MHZ in `grep "$FNAME" /proc/cpuinfo | awk "{print \\\$$FINDEX}"`; do
			CURR_MHZ=${CURR_MHZ/.*}
			if [ "$CURR_MHZ" = "" ]; then
				echo ERROR: Unable to extract CPU speed from /proc
				exit -1
			fi

			if [ $CURR_MHZ -gt $MAX_MHZ ]; then
				MAX_MHZ=$CURR_MHZ
			fi
		done
	done

	MHZ=$MAX_MHZ
	return
}

# Print help message
usage() {
	echo "tlbmiss_cost.sh [options]
options:
 --fetch-calibrator         Download and build calibrator if not in path
 --fetch-stream             Download and build STREAM if not in path
 -c, --calibrator           Path to calibrator helper if not in path
 -s, --stream               Path to STREAM helper if not in path
 -q, --quiet                Be less verbose in output
 -v, --verbose              Be more verbose in output
 -h, --help                 Print this help message"
	exit 1
}
# Print verbose message to stderr if --verbose is specified
print_trace()
{
	if [ $VERBOSE -ge 3 ]; then
		echo "TRACE: $@" 1>&2
	fi
}

print_error()
{
	if [ $VERBOSE -ge 1 ]; then
		echo "ERROR: $@" 1>&2
	fi
}

die()
{
	print_error $@
	exit -1
}

calibrator_fetch()
{
	if [ "`which calibrator`" != "" -o -e ./calibrator ]; then
		echo Calibrator is already in path or in current directory
		return
	fi

	TMPFILE=`mktemp`.c
	if [ "$TMPFILE" = "" ]; then
		die Failed to create tmpfile
	fi
	trap "rm $TMPFILE; exit" INT

	WGET=`which wget 2> /dev/null`
	if [ "$WGET" = "" ]; then
		rm $TMPFILE
		die wget is not installed, cannot fetch calibrator.c
	fi

	wget http://homepages.cwi.nl/~manegold/Calibrator/v0.9e/calibrator.c -O $TMPFILE || die Failed to download calibrator.c

	# Calibrator defines a function round() which sometimes collides with
	# a system-defined version. This patch removes the naming collision
	PATCHFILE=`basename $TMPFILE`
	echo "--- $PATCHFILE.orig	2010-02-02 14:34:38.000000000 +0000
+++ $PATCHFILE	2010-02-02 14:35:27.000000000 +0000
@@ -128,7 +128,7 @@
	exit(1);
 }

-lng round(dbl x)
+lng calibrator_round(dbl x)
 {
	return (lng)(x + 0.5);
 }
@@ -890,16 +890,16 @@
	fprintf(fp, \")\n\");
	fprintf(fp, \"set y2tics\");
	for (l = 0, s = \" (\"; l <= cache->levels; l++, s = \", \") {
-		if (!delay)	fprintf(fp, \"%s'(%ld)' %f\", s, round(CYperIt(cache->latency1[l] - delay)), NSperIt(cache->latency1[l] - delay));
-			else	fprintf(fp, \"%s'(%ld)' %f\", s, round(CYperIt(cache->latency2[l] - delay)), NSperIt(cache->latency2[l] - delay));
+		if (!delay)	fprintf(fp, \"%s'(%ld)' %f\", s, calibrator_round(CYperIt(cache->latency1[l] - delay)), NSperIt(cache->latency1[l] - delay));
+			else	fprintf(fp, \"%s'(%ld)' %f\", s, calibrator_round(CYperIt(cache->latency2[l] - delay)), NSperIt(cache->latency2[l] - delay));
	}
	for (y = 1; y <= yh; y *= 10) {
		fprintf(fp, \"%s'%1.3g' %ld\", s, (dbl)(y * MHz) / 1000.0, y);
	}
	fprintf(fp, \")\n\");
	for (l = 0; l <= cache->levels; l++) {
-		if (!delay)	z = (dbl)round(CYperIt(cache->latency1[l] - delay)) * 1000.0 / (dbl)MHz;
-			else	z = (dbl)round(CYperIt(cache->latency2[l] - delay)) * 1000.0 / (dbl)MHz;
+		if (!delay)	z = (dbl)calibrator_round(CYperIt(cache->latency1[l] - delay)) * 1000.0 / (dbl)MHz;
+			else	z = (dbl)calibrator_round(CYperIt(cache->latency2[l] - delay)) * 1000.0 / (dbl)MHz;
		fprintf(fp, \"set label %ld '(%1.3g)  ' at %f,%f right\n\", l + 1, z, xl, z);
		fprintf(fp, \"set arrow %ld from %f,%f to %f,%f nohead lt 0\n\", l + 1, xl, z, xh, z);
	}
@@ -986,16 +986,16 @@
	fprintf(fp, \"%s'<L1>' %ld)\n\", s, TLB->mincachelines);
	fprintf(fp, \"set y2tics\");
	for (l = 0, s = \" (\"; l <= TLB->levels; l++, s = \", \") {
-		if (!delay)	fprintf(fp, \"%s'(%ld)' %f\", s, round(CYperIt(TLB->latency1[l] - delay)), NSperIt(TLB->latency1[l] - delay));
-			else	fprintf(fp, \"%s'(%ld)' %f\", s, round(CYperIt(TLB->latency2[l] - delay)), NSperIt(TLB->latency2[l] - delay));
+		if (!delay)	fprintf(fp, \"%s'(%ld)' %f\", s, calibrator_round(CYperIt(TLB->latency1[l] - delay)), NSperIt(TLB->latency1[l] - delay));
+			else	fprintf(fp, \"%s'(%ld)' %f\", s, calibrator_round(CYperIt(TLB->latency2[l] - delay)), NSperIt(TLB->latency2[l] - delay));
	}
	for (y = 1; y <= yh; y *= 10) {
		fprintf(fp, \"%s'%1.3g' %ld\", s, (dbl)(y * MHz) / 1000.0, y);
	}
	fprintf(fp, \")\n\");
	for (l = 0; l <= TLB->levels; l++) {
-		if (!delay)	z = (dbl)round(CYperIt(TLB->latency1[l] - delay)) * 1000.0 / (dbl)MHz;
-			else	z = (dbl)round(CYperIt(TLB->latency2[l] - delay)) * 1000.0 / (dbl)MHz;
+		if (!delay)	z = (dbl)calibrator_round(CYperIt(TLB->latency1[l] - delay)) * 1000.0 / (dbl)MHz;
+			else	z = (dbl)calibrator_round(CYperIt(TLB->latency2[l] - delay)) * 1000.0 / (dbl)MHz;
		fprintf(fp, \"set label %ld '(%1.3g)  ' at %f,%f right\n\", l + 1, z, xl, z);
		fprintf(fp, \"set arrow %ld from %f,%f to %f,%f nohead lt 0\n\", l + 1, xl, z, xh, z);
	}
@@ -1023,9 +1023,9 @@
	FILE	*fp = stdout;

	fprintf(fp, \"CPU loop + L1 access:    \");
-	fprintf(fp, \" %6.2f ns = %3ld cy\n\", NSperIt(cache->latency1[0]), round(CYperIt(cache->latency1[0])));
+	fprintf(fp, \" %6.2f ns = %3ld cy\n\", NSperIt(cache->latency1[0]), calibrator_round(CYperIt(cache->latency1[0])));
	fprintf(fp, \"             ( delay:    \");
-	fprintf(fp, \" %6.2f ns = %3ld cy )\n\", NSperIt(delay),            round(CYperIt(delay)));
+	fprintf(fp, \" %6.2f ns = %3ld cy )\n\", NSperIt(delay),            calibrator_round(CYperIt(delay)));
	fprintf(fp, \"\n\");
	fflush(fp);
 }
@@ -1047,8 +1047,8 @@
			fprintf(fp, \" %3ld KB \", cache->size[l] / 1024);
		}
		fprintf(fp, \" %3ld bytes \", cache->linesize[l + 1]);
-		fprintf(fp, \" %6.2f ns = %3ld cy \" , NSperIt(cache->latency2[l + 1] - cache->latency2[l]), round(CYperIt(cache->latency2[l + 1] - cache->latency2[l])));
-		fprintf(fp, \" %6.2f ns = %3ld cy\n\", NSperIt(cache->latency1[l + 1] - cache->latency1[l]), round(CYperIt(cache->latency1[l + 1] - cache->latency1[l])));
+		fprintf(fp, \" %6.2f ns = %3ld cy \" , NSperIt(cache->latency2[l + 1] - cache->latency2[l]), calibrator_round(CYperIt(cache->latency2[l + 1] - cache->latency2[l])));
+		fprintf(fp, \" %6.2f ns = %3ld cy\n\", NSperIt(cache->latency1[l + 1] - cache->latency1[l]), calibrator_round(CYperIt(cache->latency1[l + 1] - cache->latency1[l])));
	}
	fprintf(fp, \"\n\");
	fflush(fp);
@@ -1075,9 +1075,9 @@
		} else {
			fprintf(fp, \"  %3ld KB  \", TLB->pagesize[l + 1] / 1024);
		}
-		fprintf(fp, \" %6.2f ns = %3ld cy \", NSperIt(TLB->latency2[l + 1] - TLB->latency2[l]), round(CYperIt(TLB->latency2[l + 1] - TLB->latency2[l])));
+		fprintf(fp, \" %6.2f ns = %3ld cy \", NSperIt(TLB->latency2[l + 1] - TLB->latency2[l]), calibrator_round(CYperIt(TLB->latency2[l + 1] - TLB->latency2[l])));
 /*
-		fprintf(fp, \" %6.2f ns = %3ld cy\" , NSperIt(TLB->latency1[l + 1] - TLB->latency1[l]), round(CYperIt(TLB->latency1[l + 1] - TLB->latency1[l])));
+		fprintf(fp, \" %6.2f ns = %3ld cy\" , NSperIt(TLB->latency1[l + 1] - TLB->latency1[l]), calibrator_round(CYperIt(TLB->latency1[l + 1] - TLB->latency1[l])));
 */
		fprintf(fp, \"\n\");
	}
" | patch -d /tmp

	LICENSE_END=`grep -n "^ \*/" $TMPFILE | head -1 | cut -f1 -d:`
	echo Displaying calibrator license
	head -$LICENSE_END $TMPFILE
	echo
	echo Calibrator is an external tool used by tlbmiss_cost.sh. The license
	echo for this software is displayed above. Are you willing to accept the
	echo -n "terms of this license [Y/N]? "
	read INPUT

	if [ "$INPUT" != "Y" -a "$INPUT" != "y" ]; then
		rm $TMPFILE
		echo Bailing...
		return
	fi
	echo Building...
	gcc $TMPFILE -w -lm -o calibrator || die Failed to compile calibrator
	echo Calibrator available at ./calibrator. For future use, run tlbmiss_cost.sh
	echo from current directory or copy calibrator into your PATH
	echo

	rm $TMPFILE
}

calibrator_calc()
{
	if [ "$CALIBRATOR" = "" ]; then
		CALIBRATOR=`which calibrator 2>/dev/null`
		if [ "$CALIBRATOR" = "" ]; then
			CALIBRATOR="./calibrator"
		fi
	fi

	if [[ ! -x $CALIBRATOR ]]; then
		die "Unable to locate calibrator. Consider using --fetch-calibrator."
	fi

	cpumhz
	SIZE=$((13*1048576))
	STRIDE=3932160
	PREFIX=tlbmiss-cost-results
	TMPFILE=`mktemp`
	TOLERANCE=2
	MATCH_REQUIREMENT=3
	MEASURED=0
	FAILED_MEASURE=0

	if [ "$TMPFILE" = "" ]; then
		die Failed to create tmpfile
	fi
	if [ "$MHZ" = "" ]; then
		die Failed to calculate CPU MHz
	fi
	trap "rm $TMPFILE*; exit" INT

	MATCHED=0
	LAST_LATENCY_CYCLES=-1

	print_trace Beginning TLB measurement using calibrator
	print_trace Measured CPU Speed: $MHZ MHz
	print_trace Starting Working Set Size \(WSS\): $SIZE bytes
	print_trace Required tolerance for match: $MATCH_REQUIREMENT cycles

	# Keep increasing size until TLB latency is being measured consistently
	while [ $MATCHED -lt $MATCH_REQUIREMENT ]; do
		$CALIBRATOR $MHZ $SIZE $PREFIX > $TMPFILE 2>&1
		if [ $? != 0 ]; then
			SIZE=$(($SIZE*2))
			continue
		fi

		LATENCY_CYCLES=`grep ^TLBs: -A 2 $TMPFILE | tail -1 | awk -F = '{print $2}'`
		LATENCY_CYCLES=`echo $LATENCY_CYCLES | awk '{print $1}'`

		if [ "$LATENCY_CYCLES" = "" ]; then
			FAILED_MEASURE=$(($FAILED_MEASURE+1))
			if [ $MEASURED -eq 0 ]; then
				SIZE=$(($SIZE*3/2))
				FAILED_MEASURE=0
			else
				if [ $FAILED_MEASURE -eq 3 ]; then
					SIZE=$(($SIZE+$STRIDE))
					FAILED_MEASURE=0
					print_trace No TLB Latency measured: New WSS $SIZE
				else
					print_trace No TLB Latency measured: Retrying
				fi
			fi
			continue
		fi
		LOW_TOLERANCE=$(($LATENCY_CYCLES-$TOLERANCE))
		HIGH_TOLERANCE=$(($LATENCY_CYCLES+$TOLERANCE))
		if [ $LAST_LATENCY_CYCLES -ge $LOW_TOLERANCE -a \
				$LAST_LATENCY_CYCLES -le $HIGH_TOLERANCE ]; then
			MATCHED=$(($MATCHED+1))
			print_trace Measured TLB Latency $LATENCY_CYCLES cycles within tolerance. Matched $MATCHED/$MATCH_REQUIREMENT
		else
			if [ $LAST_LATENCY_CYCLES -ne -1 ]; then
				print_trace Measured TLB Latency $LATENCY_CYCLES cycles outside tolerance
			fi
			MATCHED=0
		fi

		LAST_LATENCY_CYCLES=$LATENCY_CYCLES
		SIZE=$(($SIZE+$STRIDE))
		MEASURED=$(($MEASURED+1))
		FAILED_MEASURE=0
	done
	rm $TMPFILE*
	rm tlbmiss-cost-results*
}

# This method uses the stream memory benchmark which can be found here:
# http://www.cs.virginia.edu/stream/FTP/Code/stream.c
# and should be compiled with this command line:
# gcc -m32 -O3 -DN=44739240 stream.c -o STREAM
# and then placed in the same directory as this script

stream_fetch()
{
	# STREAM binary is in caps as there is a commonly-available binary
	# called stream that is packaged with ImageMagick. This avoids some
	# confusion
	if [ "`which STREAM`" != "" -o -e ./STREAM ]; then
		echo STREAM is already in path or in current directory
		return
	fi

	TMPFILE=`mktemp`.c
	if [ "$TMPFILE" = "" ]; then
		die Failed to create tmpfile
	fi
	trap "rm $TMPFILE; exit" INT

	WGET=`which wget 2> /dev/null`
	if [ "$WGET" = "" ]; then
		rm $TMPFILE
		die wget is not installed, cannot fetch stream.c
	fi

	wget http://www.cs.virginia.edu/stream/FTP/Code/stream.c -O $TMPFILE || die Failed to download stream.c

	LICENSE_END=`grep -n "^/\*--" $TMPFILE | tail -1 | cut -f1 -d:`
	echo Displaying STREAM license
	head -$LICENSE_END $TMPFILE
	echo
	echo STREAM is an external tool used by tlbmiss_cost.sh. The license
	echo for this software is displayed above. Are you willing to accept the
	echo -n "terms of this license [Y/N]? "
	read INPUT

	if [ "$INPUT" != "Y" -a "$INPUT" != "y" ]; then
		rm $TMPFILE
		echo Bailing...
		return
	fi
	echo Building...
	gcc -m32 -O3 -w -DN=44739240 $TMPFILE -o STREAM || die Failed to compile STREAM
	echo STREAM is available at ./STREAM. For future use, run tlbmiss_cost.sh
	echo from current directory or copy STREAM into your PATH
	echo

	rm $TMPFILE
}

seperate_dtlb_pagewalk_groups()
{
	TIMER_DTLB_EVENT=`oprofile_map_events.pl --event timer | cut -d: -f1 2> /dev/null`
	TIMER_WALK_EVENT=`oprofile_map_events.pl --event timer30 | cut -d: -f1 2> /dev/null`

	# Help debug problems launching oprofile
	print_trace oprofile launch commands as follows
	print_trace dtlb misses :: oprofile_start --event timer --event dtlb_miss --sample-cycle-factor 5
	print_trace tablewalk cycles :: oprofile_start --event timer30 --event tablewalk_cycles --sample-cycle-factor 5 --sample-event-factor $SAMPLE_EVENT_FACTOR

	print_trace Rerunning benchmark to measure number of DTLB misses
	$OPST $VMLINUX --event timer --event dtlb_miss --sample-cycle-factor 5 >/dev/null 2>&1 || \
		die "Error starting oprofile, check oprofile_map_event.pl for appropriate timer and dtlb_miss events."
	$STREAM >/dev/null 2>&1

	opcontrol --stop >/dev/null 2>&1
	opcontrol --dump >/dev/null 2>&1

	# First ensure that the location of event counters are where we
	# expect them to be. The expectation is that the timer30 is in
	# the first column and the tablewalk_cycles is in the third
	SAMPLES_START=`opreport | grep -n "samples|" | head -1 | cut -d: -f1`
	if [ "$SAMPLES_START" = "" ]; then
		die Could not establish start of samples from opreport
		SAMPLES_START=$(($COUNT_START+1))
	fi
	INDEX=`opreport | head -$SAMPLES_START | grep "^Counted .* events" | grep -n $TIMER_DTLB_EVENT | cut -d: -f1`
	TIMER_DTLB_FIELD=$((1+2*($INDEX - 1)))
	if [ $TIMER_DTLB_FIELD -eq 1 ]; then
		DTLB_TIMER_INDEX=1
	else
		DTLB_TIMER_INDEX=2
	fi

	TIMER_DTLB_SCALE=`opreport | grep "$TIMER_DTLB_EVENT" | head -1 | sed 's/.* count \([0-9]*\).*/\1/'`
	DTLB_SCALE=`opreport | grep "$DTLB_EVENT" | head -1 | sed 's/.* count \([0-9]*\).*/\1/'`

	RESULTS=`opreport | grep " STREAM" | head -1`
	FIELD1=`echo "$RESULTS" | sed 's/[[:space:]]*\([0-9]*\).*/\1/'`
	FIELD2=`echo "$RESULTS" | sed 's/[[:space:]]*[0-9]*[[:space:]]*[[:graph:]]*[[:space:]]*\([0-9]*\).*/\1/'`
	RESULTS=`opreport | grep " vmlinux" | head -1`
	KERNEL_FIELD1=`echo "$RESULTS" | sed 's/[[:space:]]*\([0-9]*\).*/\1/'`
	KERNEL_FIELD2=`echo "$RESULTS" | sed 's/[[:space:]]*[0-9]*[[:space:]]*[[:graph:]]*[[:space:]]*\([0-9]*\).*/\1/'`

	if [ $DTLB_TIMER_INDEX -eq 1 ] ; then
		TIMER_DTLB=$(($FIELD1+$KERNEL_FIELD1))
		DTLB=$(($FIELD2+$KERNEL_FIELD2))
	else
		TIMER_DTLB=$(($FIELD2+$KERNEL_FIELD2))
		DTLB=$(($FIELD1+$KERNEL_FIELD1))
	fi

	print_trace Shutting down oprofile
	opcontrol --shutdown >/dev/null 2>&1
	opcontrol --deinit >/dev/null 2>&1

	# Next STREAM needs to be run measuring the tablewalk_cycles. Because
	# of differences in the frequency CPU events occur, there are
	# alterations in the timing. To make an accurate comparison, the
	# cycle counts of the two profiles need to be very similar. oprofile
	# does not give much help here in matching up different reports taking
	# different readings so there is nothing really to do but run STREAM
	# multiple times, scaling the events at different rates until a
	# reasonably close match is found.

	# The cycle counts for two oprofiles must be within 10% of each other
	TOLERANCE=$(($TIMER_DTLB*4/100))
	SAMPLE_EVENT_FACTOR=1
	LOW_TIMER_WALK=0
	HIGH_TIMER_WALK=0

	print_trace Running benchmark to measure table walk cycles
	while [ $TIMER_DTLB -ge $LOW_TIMER_WALK -a $TIMER_DTLB -ge $HIGH_TIMER_WALK ]; do

		if [ $LOW_TIMER_WALK -ne 0 ]; then
			print_trace High diff with scaling x$LAST_SAMPLE_EVENT_FACTOR. Required $TIMER_DTLB +/ $TOLERANCE, got $TIMER_WALK
		fi

		$OPST $VMLINUX --event timer30 --event tablewalk_cycles --sample-cycle-factor 5 --sample-event-factor $SAMPLE_EVENT_FACTOR >/dev/null 2>&1 || \
			die "Error starting oprofile, check oprofile_map_event.pl for appropriate timer30 and tablewalk_cycles events."
		$STREAM >/dev/null 2>&1

		opcontrol --stop >/dev/null 2>&1
		opcontrol --dump >/dev/null 2>&1

		# Extract the event counts
		TIMER_WALK_SCALE=`opreport | grep "$TIMER_WALK_EVENT" | head -1 | sed 's/.* count \([0-9]*\).*/\1/'`
		WALK_SCALE=`opreport | grep "$WALK_EVENT" | head -1 | sed 's/.* count \([0-9]*\).*/\1/'`

		# This shouldn't happen. One would expect that the minimum sample
		# rate for any of the timers in any groups is the same. If they
		# differ, it might be a simple bug in oprofile_map_event that
		# needs fixing. In the event this bug is reported, get the CPU
		# type and the output of opcontrol --list-events
		if [ $TIMER_DTLB_SCALE -ne $TIMER_WALK_SCALE ]; then
			die Cycle CPUs were sampled at different rates.
		fi

		RESULTS=`opreport | grep " STREAM" | head -1`
		FIELD1=`echo "$RESULTS" | sed 's/[[:space:]]*\([0-9]*\).*/\1/'`
		FIELD2=`echo "$RESULTS" | sed 's/[[:space:]]*[0-9]*[[:space:]]*[[:graph:]]*[[:space:]]*\([0-9]*\).*/\1/'`
		RESULTS=`opreport | grep " vmlinux" | head -1`
		KERNEL_FIELD1=`echo "$RESULTS" | sed 's/[[:space:]]*\([0-9]*\).*/\1/'`
		KERNEL_FIELD2=`echo "$RESULTS" | sed 's/[[:space:]]*[0-9]*[[:space:]]*[[:graph:]]*[[:space:]]*\([0-9]*\).*/\1/'`

		if [ $DTLB_TIMER_INDEX -eq 1 ] ; then
			TIMER_WALK=$(($FIELD1+$KERNEL_FIELD1))
			WALK=$(($FIELD2+$KERNEL_FIELD2))
		else
			TIMER_WALK=$(($FIELD2+$KERNEL_FIELD2))
			WALK=$(($FIELD1+$KERNEL_FIELD1))
		fi

		LOW_TIMER_WALK=$(($TIMER_WALK-$TOLERANCE))
		HIGH_TIMER_WALK=$(($TIMER_WALK+$TOLERANCE))

		# Scale faster if the difference between timers is huge
		LAST_SAMPLE_EVENT_FACTOR=$SAMPLE_EVENT_FACTOR
		if [ $(($TIMER_DTLB*3/4-$HIGH_TIMER_WALK)) -gt 0 ]; then
			SAMPLE_EVENT_FACTOR=$(($SAMPLE_EVENT_FACTOR+3))
		elif [ $(($TIMER_DTLB*9/10-$HIGH_TIMER_WALK)) -gt 0 ]; then
			SAMPLE_EVENT_FACTOR=$(($SAMPLE_EVENT_FACTOR+2))
		else
			SAMPLE_EVENT_FACTOR=$(($SAMPLE_EVENT_FACTOR+1))
		fi

		opcontrol --shutdown >/dev/null 2>&1
		opcontrol --deinit >/dev/null 2>&1
	done

	print_trace "DTLB       Scale: $DTLB_SCALE"
	print_trace "Walk       Scale: $WALK_SCALE"
	print_trace "DTLB    events: $DTLB"
	print_trace "Walk    events: $WALK"
	print_trace "Cycle DTLB Scale: $TIMER_DTLB_SCALE"
	print_trace "Cycle Walk Scale: $TIMER_WALK_SCALE"
	print_trace "Cycle DTLB events: $TIMER_DTLB"
	print_trace "Cycle Walk events: $TIMER_WALK"
}

dtlb_pagewalk_same_group()
{
	print_trace oprofile launch command as follows
	print_trace $OPST --event dtlb_miss --event tablewalk_cycles

	$OPST $VMLINUX --event dtlb_miss --event tablewalk_cycles > /dev/null 2>&1 || \
		die "Error starting oprofile, check oprofile_map_event.pl for appropriate dtlb_miss and tablewalk_cycles events."
	$STREAM >/dev/null 2>&1

	opcontrol --stop >/dev/null 2>&1
	opcontrol --dump >/dev/null 2>&1

	# First ensure that the location of event counters are where we
	# expect them to be. The expectation is that tablewalk_cycles is in
	# the first column and the dtlb_misses is in the second
	SAMPLES_START=`opreport | grep -n "samples|" | head -1 | cut -d: -f1`
	if [ "$SAMPLES_START" = "" ]; then
		die Could not establish start of samples from opreport
	fi
	INDEX=`opreport | head -$SAMPLES_START | grep "^Counted .* events" | grep -n $WALK_EVENT | cut -d: -f1`
	WALK_FIELD=$((1+2*($INDEX - 1)))
	if [ $WALK_FIELD -ne 1 ]; then
		die Table walk events are not in the expected column, parse failure
	fi
	INDEX=`opreport | head -$SAMPLES_START | grep "^Counted .* events" | grep -n $DTLB_EVENT | cut -d: -f1`
	DTLB_FIELD=$((1+2*($INDEX - 1)))
	if [ $DTLB_FIELD -ne 3 ]; then
		die DTLB miss events are not in the expected column, parse failure
	fi

	# Columns look ok, extract the event counts
	DTLB_SCALE=`opreport | grep "$DTLB_EVENT" | head -1 | sed 's/.* count \([0-9]*\).*/\1/'`
	WALK_SCALE=`opreport | grep "$WALK_EVENT" | head -1 | sed 's/.* count \([0-9]*\).*/\1/'`
	RESULTS=`opreport | grep " STREAM" | head -1`
	WALK=`echo "$RESULTS" | sed 's/[[:space:]]*\([0-9]*\).*/\1/'`
	DTLB=`echo "$RESULTS" | sed 's/[[:space:]]*[0-9]*[[:space:]]*[[:graph:]]*[[:space:]]*\([0-9]*\).*/\1/'`
	RESULTS=`opreport | grep " vmlinux" | head -1`
	KERN_TABLE_WALK=`echo "$RESULTS" | sed 's/[[:space:]]*\([0-9]*\).*/\1/'`
	KERN_TLB_MISS=`echo "$RESULTS" | sed 's/[[:space:]]*[0-9]*[[:space:]]*[[:graph:]]*[[:space:]]*\([0-9]*\).*/\1/'`

	print_trace "DTLB       Scale: $DTLB_SCALE"
	print_trace "Walk       Scale: $WALK_SCALE"
	print_trace "DTLB    events: $DTLB + $KERN_TLB_MISS = $(($DTLB+$KERN_TLB_MISS))"
	print_trace "Walk    events: $WALK + $KERN_TABLE_WALK = $(($WALK+$KERN_TABLE_WALK))"

	if [[ "$KERN_TLB_MISS" != "" ]]; then
		DTLB=$(($DTLB+$KERN_TLB_MISS))
	fi
	if [[ "$KERN_TABLE_WALK" != "" ]]; then
		WALK=$(($WALK+$KERN_TABLE_WALK))
	fi

	opcontrol --shutdown >/dev/null 2>&1
	opcontrol --deinit >/dev/null 2>&1
}

oprofile_calc()
{
	if [ "$STREAM" = "" ]; then
		STREAM="./STREAM"
	fi

	if [[ ! -x $STREAM ]]; then
		die "Unable to locate STREAM. Consider using --fetch-stream."
	fi

	OPST=`which oprofile_start.sh`
	if [ "$OPST" = "" ]; then
		OPST="../oprofile_start.sh"
	fi

	if [[ ! -x $OPST ]]; then
		die "Unable to locate oprofile_start.sh."
	fi

	print_trace Forcing shutdown of oprofile
	opcontrol --shutdown >/dev/null 2>&1
	opcontrol --deinit >/dev/null 2>&1

	print_trace Gathering the name of CPU events
	WALK_EVENT=`oprofile_map_events.pl --event tablewalk_cycles | cut -d: -f1 2> /dev/null`
	DTLB_EVENT=`oprofile_map_events.pl --event dtlb_miss | cut -d: -f1 2> /dev/null`

	GROUP1=`echo $WALK_EVENT | sed 's/.*\(GRP[0-9]*\)/\1/'`
	GROUP2=`echo $DTLB_EVENT | sed 's/.*\(GRP[0-9]*\)/\1/'`

	print_trace Warming the benchmark to avoid page faults of the binary
	$STREAM >/dev/null 2>&1

	if [[ "$GROUP1" == "$GROUP2" ]] ; then
		print_trace "Events are in the same group: $GROUP1, using one oprofile pass"
		dtlb_pagewalk_same_group
	else
		print_trace "Events are in different groups: $GROUP1 and $GROUP2, using multiple oprofile passes"
		seperate_dtlb_pagewalk_groups
	fi

	WALK=$(($WALK*$WALK_SCALE))
	DTLB=$(($DTLB*$DTLB_SCALE))
	LAST_LATENCY_CYCLES=$(($WALK/$DTLB))
}

ARGS=`getopt -o c:s:vqh --long calibrator:,stream:,vmlinux:,verbose,quiet,fetch-calibrator,fetch-stream,help -n 'tlbmiss_cost.sh' -- "$@"`

eval set -- "$ARGS"

while true ; do
	case "$1" in
		-c|--calibrator) CALIBRATOR="$2" ; shift 2 ;;
		-s|--stream) STREAM="$2" ; shift 2 ;;
		--vmlinux) VMLINUX="--vmlinux $2" ; shift 2 ;;
		-v|--verbose) VERBOSE=$(($VERBOSE+1)); shift;;
		-q|--quiet) VERBOSE=$(($VERBOSE-1)); shift;;
		--fetch-calibrator) calibrator_fetch; shift;;
		--fetch-stream) stream_fetch; shift;;
		-h|--help) usage; shift;;
		"") shift ; break ;;
		"--") shift ; break ;;
		*) die "Unrecognized option $1" ;;
	esac
done

ARCH=`uname -m | sed -e s/i.86/i386/`

if [[ "$ARCH" == "ppc64" || "$ARCH" == "ppc" ]]; then
	oprofile_calc
else
	calibrator_calc
fi

echo TLB_MISS_COST=$LAST_LATENCY_CYCLES
exit 0
