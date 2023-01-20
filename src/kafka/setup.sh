#!/bin/sh
#SBATCH --account=def-jacobsen
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32                             # number of MPI processes
#SBATCH --time=1:00:00                                  # time (DD-HH:MM)
#SBATCH --job-name=jobkafka1
#SBATCH --mem=20G
#SBATCH --mail-user=michael.dangana@mail.utoronto.ca              # notification for job conditions
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
 

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
    #SCRIPT=`readlink -f $0`
    #echo SCRIPT=$SCRIPT
    #BASE=src/kafka/$(basename -- $SCRIPT)
    BASE=~/scratch/kfpca/
    CFG_FILE="${BASE}config/testbed.properties"
    for x in `grep = $CFG_FILE`; do
        CFG["${x//=*/}"]="${x//*=/}"
    done
    echo config file = $CFG_FILE, properties = ${!CFG[@]}
  
    KAFKA_HOME=${CFG["kafka.home"]}
    export KAFKA_HOME=`echo $KAFKA_HOME | sed 's/\(.*\/\)/\1/'`;
    #export KAFKA_HEAP_OPTS="-Xmx64m"
    #export ZK_SERVER_HEAP=512
    export CFG BASE NUM_BROKERS=${CFG["kafka.brokers"]}
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
	     sed -i 's/#\(listeners=PLAINTEXT.*\)9092/\1'$((9091+$i))'/' $CFG_FILE
	     sed -i 's/\(log.dirs=.*\)/\/tmp\/kafka-logs-'$i'/' $CFG_FILE
	     #sed -i 's/\(zookeeper.connect=.*:\).*/\1'$((2181+$i))'/' $CFG_FILE
	     echo Set up broker ${KAFKA_HOME}_broker$i
	 done
    #fi
}


launch_broker() {
    i=$1
    echo; echo Launching broker $i...
    CFG_FILE=${KAFKA_HOME}_broker$i/config/server.properties
    sed -i 's/\(queued.max.requests*=\).*/\1'$QUEUE_REQ'/' $CFG_FILE
    ${KAFKA_HOME}_broker$i/bin/kafka-server-stop.sh 
    sleep $STARTUP_TS
    if $START; then
        nohup ${KAFKA_HOME}_broker$i/bin/kafka-server-start.sh $CFG_FILE > \
	    broker.log 2>&1 &
        sleep $STARTUP_TS
    fi
}


launch_brokers() {
    echo; echo Launching brokers...
    echo start = $START, startup_ts = $STARTUP_TS
    ${KAFKA_HOME}/bin/zookeeper-server-stop.sh
    if $START; then
        sleep $STARTUP_TS
        nohup ${KAFKA_HOME}/bin/zookeeper-server-start.sh $ZCFG > zoo.log 2>&1 &
    fi
    for i in $(seq 1 $NUM_BROKERS); do
        launch_broker $i
    done
    if $START; then
      topic=${CFG["kafka.topics"]//,*/}
      nohup $KAFKA_HOME/bin/kafka-topics.sh --create \
        --bootstrap-server localhost:9092 \ #--zookeeper localhost:2181 \
        --replication-factor $NUM_BROKERS --partitions 10 --topic $topic \
	> topics.log 2>&1 &
      #$KAFKA_HOME/bin/kafka-topics.sh --describe --topic $topic \
       # --bootstrap-server localhost:9092 
        #--zookeeper localhost:2181
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

launch_producers_consumers() {
    echo; echo Launching consumers...
    kill -9 `cat ${BASE}consumer.pid`
    kill -9 `cat ${BASE}producer.pid`
    if $START; then
        nohup python ${BASE}src/kafka/consumer.py >& consumer.log &
	echo $! > ${BASE}consumer.pid
        sleep $STARTUP_TS
        nohup python ${BASE}src/kafka/producer.py >& producer.log &
	echo $! > ${BASE}producer.pid
    fi
}


#install_java
get_opts $@
get_properties
setup_brokers
launch_brokers
#launch_tracker
launch_producers_consumers
sleep 120

echo; echo $0 done, output in ${CFG["data.tracker.file"]}.
