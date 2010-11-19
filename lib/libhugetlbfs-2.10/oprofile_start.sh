#!/bin/bash
# Script to start oprofile

usage() {
	echo "oprofile_start.sh (c) Mel Gorman 2008"
	echo This script starts the oprofile daemon
	echo
	echo "Usage: oprofile_start.sh [options]"
	echo "    --event               High-level oprofile event to track"
	echo "    --vmlinux             Path to vmlinux"
	echo "    --sample-cycle-factor Factor which to slow down CPU cycle sampling by"
	echo "    --sample-event-factor Factor which to slow down event sampling by"
	echo "    --systemmap           Guess"
	echo "    -h, --help            Print this help message"
	echo
	exit
}

# Parse command-line arguements
SCRIPTROOT=`echo $0 | sed -e 's/oprofile_start.sh$//' | sed -e 's/^\.\///'`
EVENT=default
VMLINUX=/boot/vmlinux-`uname -r`
SYSTEMMAP=/boot/System.map-`uname -r`
FACTOR=
export PATH=$SCRIPTROOT:$PATH
ARGS=`getopt -o h --long help,event:,vmlinux:,systemmap:,sample-event-factor:,sample-cycle-factor: -n oprofile_start.sh -- "$@"`

# Cycle through arguements
eval set -- "$ARGS"
while true ; do
  case "$1" in
	--event)               EVENTS="$EVENTS $2"; shift 2;;
	--vmlinux)             VMLINUX=$2; shift 2;;
	--sample-cycle-factor) CYCLE_FACTOR="--sample-cycle-factor $2"; shift 2;;
	--sample-event-factor) EVENT_FACTOR="--sample-event-factor $2"; shift 2;;
	--systemmap)           SYSTEMMAP=$2; shift 2;;
        -h|--help) usage;;
        *) shift 1; break;;
  esac
done

# Map the events
for EVENT in $EVENTS; do
	LOWLEVEL_EVENT="$LOWLEVEL_EVENT --event `oprofile_map_events.pl $EVENT_FACTOR $CYCLE_FACTOR --event $EVENT`"
	if [ $? -ne 0 ]; then
		echo Failed to map event $EVENT to low-level oprofile event. Verbose output follows
		oprofile_map_events.pl --event $EVENT --verbose
		exit -1
	fi
done

# Check vmlinux file exists
if [ "$VMLINUX" = "" -o ! -e $VMLINUX ]; then
	echo vmlinux file \"$VMLINUX\" does not exist
	exit -1
fi

echo Stage 1: Shutting down if running and resetting
bash opcontrol --reset
bash opcontrol --stop
bash opcontrol --reset
bash opcontrol --deinit
echo

# Setup the profiler
echo Stage 2: Setting up oprofile
echo High-level event: $EVENTS
echo Low-level event: `echo $LOWLEVEL_EVENT | sed -e 's/--event //'`
echo vmlinux: $VMLINUX
echo opcontrol --setup $LOWLEVEL_EVENT --vmlinux=$VMLINUX
bash opcontrol --setup $LOWLEVEL_EVENT --vmlinux=$VMLINUX
if [ $? -ne 0 ]; then
	echo opcontrol --setup returned failed
	exit -1
fi

# Start the profiler
echo Stage 3: Starting profiler
bash opcontrol --start
if [ $? -ne 0 ]; then
	echo opcontrol --start returned failure
	exit -1
fi

exit 0
