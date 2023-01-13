#!/bin/sh

declare -A CFG=()

get_properties() {
    SCRIPT=`pwd`/$0
    BASE=src/$(basename -- $SCRIPT)
    CFG_FILE="${SCRIPT//$BASE/}config/testbed.properties"
    echo Config file is $CFG_FILE

    for x in `grep = $CFG_FILE`; do
        CFG["${x//=*/}"]="${x//*=/}"
    done
    echo properties = ${!CFG[@]}
  
    KAFKA_HOME=${CFG["kafka.home"]}
    KAFKA_HOME=`echo $KAFKA_HOME | sed 's/\(.*\)\//\1/'`;
}

setup_brokers {
    NUM_BROKERS=${CFG["kafka.num_brokers"]}
    CFG_FILE=$KAFKA_HOME_broker$i/config/server.properties
    #sed -i 's/broker.id=0/broker.id=$i/' $CFG_FILE
    #if [$(($NUM_BROKERS)) -gt 1]; then
	 for i in {0..$(($NUM_BROKERS-1))}; do
	     cp -r $KAFKA_HOME "$KAFKA_HOME_broker$i"
	     mkdir -p /tmp/kafka-logs-$i
	     sed -i 's/broker.id=0/broker.id=$i/' $CFG_FILE
	     sed -i 's/#(listeners=PLAINTEXT.*)9092/\1$((9092+$i))/' $CFG_FILE
	     sed -i 's/(log.dirs=.*)/\1-$i/' $CFG_FILE
	     sed -i 's/(zookeeper.connect=.*:).*/\1$((2181+$i))/' $CFG_FILE
	 done
    #fi
}

launch_brokers {
    get_properties
    NUM_BROKERS=${CFG["kafka.num_brokers"]}
    QUEUE_REQ=${CFG["queued.max.requests"]}
    STARTUP_TS=${CFG["kafka.startup_time_sec"]}
    for i in {0..$(($NUM_BROKERS-1))}; do
        CFG_FILE=$KAFKA_HOME_broker$i/config/server.properties
	sed -i 's/(queued.max.requests*=).*/\1$QUEUE_REQ/' $CFG_FILE
        $KAFKA_HOME_broker$i/bin/kafka-server-stop.sh 
	sleep $STARTUP_TS
        $KAFKA_HOME_broker$i/bin/kafka-server-start.sh \
	    $KAFKA_HOME_broker$i/config/server.properties
	sleep $STARTUP_TS
    done
    topic=${CFG["kafka.topics"]//,*/}
    $KAFKA_HOME/bin/kafka-topics.sh --create --zookeeper localhost:2181 \
        --replication-factor $NUM_BROKERS --partitions 10 --topic $topic
    $KAFKA_HOME/bin/kafka-topics.sh --describe --topic $topic \
        --zookeeper localhost:2181
}

get_properties
setup_brokers
launch_brokers
