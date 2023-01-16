#!/bin/sh

declare -A CFG=()


get_opts() {
    START=true
    if [[ "${@}" =~ "-h" ]] || [[ "${@}" =~ "--help" ]]; then
        echo "Usage: $0 [--start|--stop] [-h|--help]"; exit 0;
    fi
    if  [[ "${@}" =~ "--stop" ]] ; then
        START=false
    fi
    export START
}


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
    export KAFKA_HEAP_OPTS="-Xmx512m"
    export CFG BASE=${SCRIPT//$BASE/}
    export NUM_BROKERS=${CFG["kafka.brokers"]}
    export QUEUE_REQ=${CFG["queued.max.requests"]}
    export STARTUP_TS=${CFG["kafka.startup_time_sec"]}
    export ZCFG=${KAFKA_HOME}/config/zookeeper.properties
    echo KAFKA_HOME=$KAFKA_HOME
}

install_java() {
    JRE_ARCHIVE=jre-8u351-linux.tar.gz
    JRE_URL=https://javadl.oracle.com/webapps/download/AutoDL?BundleId=247127_10e8cce67c7843478f41411b7003171c
    wget $JRE_URL -O $JRE_ARCHIVE
    $tar xvf $JRE_ARCHIVE
    export JAVA_HOME=`pwd`/jre1.8.0_351
    echo PATH=$JAVA_HOME/bin:$PATH
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
	     sed -i 's/broker.id=0/broker.id='$(($i))'/' $CFG_FILE
	     sed -i 's/#\(listeners=PLAINTEXT.*\)9092/\1'$((9092+$i))'/' $CFG_FILE
	     sed -i 's/\(log.dirs=.*\)/\1-'$i'/' $CFG_FILE
	     sed -i 's/\(zookeeper.connect=.*:\).*/\1'$((2181+$i))'/' $CFG_FILE
	     echo Set up broker ${KAFKA_HOME}_broker$i
	 done
    #fi
}


launch_broker() {
    i=$1
    CFG_FILE=${KAFKA_HOME}_broker$i/config/server.properties
    sed -i 's/\(queued.max.requests*=\).*/\1'$QUEUE_REQ'/' $CFG_FILE
    ${KAFKA_HOME}_broker$i/bin/kafka-server-stop.sh 
    sleep $STARTUP_TS
    if $START; then
        ${KAFKA_HOME}_broker$i/bin/kafka-server-start.sh $CFG_FILE
        sleep $STARTUP_TS
    fi
}


launch_brokers() {
    echo; echo Launching brokers...
    ${KAFKA_HOME}/bin/zookeeper-server-stop.sh
    if $START; then
        sleep $STARTUP_TS
        nohup ${KAFKA_HOME}/bin/zookeeper-server-start.sh $ZCFG > zoo.log 2>&1
    fi
    for i in $(seq 1 $NUM_BROKERS); do
        launch_broker $i
    done
    if $START; then
      topic=${CFG["kafka.topics"]//,*/}
      $KAFKA_HOME/bin/kafka-topics.sh --create --zookeeper localhost:2181 \
        --replication-factor $NUM_BROKERS --partitions 10 --topic $topic
      $KAFKA_HOME/bin/kafka-topics.sh --describe --topic $topic \
        --zookeeper localhost:2181
    fi
}

launch_tracker() {
    echo; echo Launching tracker... 
    export FLASK_APP=${BASE}src/kafka/tracker.py
    kill -9 `cat ${BASE}flask.pid`
    if $START; then
        nohup python -m flask run >& nohup.out & 
        echo $! > ${BASE}flask.pid
    fi
}

#install_java
get_opts $@
get_properties
setup_brokers
launch_brokers
launch_tracker

echo; echo $0 done.
