#!/bin/sh


SCRIPT=`pwd`/$0

BASE=src/$(basename -- $SCRIPT)

CFG_FILE="${SCRIPT//$BASE/}config/testbed.properties"

echo Config file is $CFG_FILE

declare -A PROPS=()

for x in `grep = $CFG_FILE`; do
    PROPS["${x//=*/}"]="${x//*=/}"
done

echo props=${!PROPS[@]}
