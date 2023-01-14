#!/bin/sh

declare -A CFG=()

get_properties() {
    echo; echo Retrieving properties...
    SCRIPT=`readlink -f $0`
    echo SCRIPT=$SCRIPT
    BASE=src/kafka/$(basename -- $SCRIPT)
    CFG_FILE="${SCRIPT//$BASE/}config/testbed.properties"

    for x in `grep = $CFG_FILE`; do
        CFG["${x//=*/}"]="${x//*=/}"
    done
    echo config file = $CFG_FILE, properties = ${!CFG[@]}
  
    KAFKA_HOME=${CFG["kafka.home"]}
    export KAFKA_HOME=`echo $KAFKA_HOME | sed 's/\(.*\/\)/\1/'`;
    export CFG BASE=${SCRIPT//$BASE/}
    echo KAFKA_HOME=$KAFKA_HOME
}

setup_brokers() {
    echo; echo Setting up brokers...
    NUM_BROKERS=${CFG["kafka.brokers"]}
    echo NUM_BROKERS=$NUM_BROKERS, CFG_FILE=$CFG_FILE
    #sed -i 's/broker.id=0/broker.id=$i/' $CFG_FILE
    #if [$(($NUM_BROKERS)) -gt 1]; then
	 for i in $(seq 1 $NUM_BROKERS); do
             CFG_FILE=${KAFKA_HOME}_broker$i/config/server.properties
	     cp -r $KAFKA_HOME "${KAFKA_HOME}_broker$i"
	     mkdir -p /tmp/kafka-logs-$i
	     sed -i 's/broker.id=0/broker.id=$i/' $CFG_FILE
	     sed -i 's/#\(listeners=PLAINTEXT.*\)9092/\1$((9092+$i))/' $CFG_FILE
	     sed -i 's/\(log.dirs=.*\)/\1-$i/' $CFG_FILE
	     sed -i 's/\(zookeeper.connect=.*:\).*/\1$((2181+$i))/' $CFG_FILE
	     echo Set up broker ${KAFKA_HOME}_broker$i
	 done
    #fi
}

launch_brokers() {
    echo; echo Launching brokers...
    NUM_BROKERS=${CFG["kafka.brokers"]}
    QUEUE_REQ=${CFG["queued.max.requests"]}
    STARTUP_TS=${CFG["kafka.startup_time_sec"]}
    for i in $(seq 1 $NUM_BROKERS); do
        CFG_FILE=${KAFKA_HOME}_broker$i/config/server.properties
	sed -i 's/\(queued.max.requests*=\).*/\1$QUEUE_REQ/' $CFG_FILE
        ${KAFKA_HOME}_broker$i/bin/kafka-server-stop.sh 
	sleep $STARTUP_TS
        ${KAFKA_HOME}_broker$i/bin/kafka-server-start.sh \
	    ${KAFKA_HOME}_broker$i/config/server.properties
	sleep $STARTUP_TS
    done
    topic=${CFG["kafka.topics"]//,*/}
    $KAFKA_HOME/bin/kafka-topics.sh --create --zookeeper localhost:2181 \
        --replication-factor $NUM_BROKERS --partitions 10 --topic $topic
    $KAFKA_HOME/bin/kafka-topics.sh --describe --topic $topic \
        --zookeeper localhost:2181
}

launch_tracker() {
    echo; echo Launching tracker... 
    export FLASK_APP=${BASE}src/kafka/tracker.py
    nohup python -m flask run >& nohup.out & 
    echo $! > ${BASE}flask.pid
}

get_properties
setup_brokers
launch_brokers
launch_tracker

echo; echo $0 done.
