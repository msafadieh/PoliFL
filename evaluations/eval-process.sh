#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied."
    exit 1
fi

process=$1
output=$2
sleep_time=1

echo "Evaluation of process '$1' started."
echo "Use Ctrl-C to stop."

timestamp() {
  date "+%d/%m/%y %H:%M:%S.%N"
}

utilization_ps() {
    ps -p $process -o %cpu,%mem,cmd --no-headers
}

utilization_top() {
    top -p $process -b -n 1 | grep $process | sed -e 's/\s\+/,/g'
}

# if file exists, delete
if [ -f $output ] ; then
    rm $output
fi

# Write header
echo "Timestamp,PID,USER,PR,NI,VIRT,RES,SHR,S,%CPU,%MEM,TIME+,COMMAND" >> $output

while true
do
    echo "$(timestamp),$(utilization_top)" >> $output
    sleep $sleep_time
done
