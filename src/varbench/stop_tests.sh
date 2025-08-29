

TEST_TYPE="HPA" # HPA|KF|SIMPLE

for w in 2; do
    hpa=kafka-worker${w}-hpa
    host=ksurf-worker${w}
    if [ $TEST_TYPE = "HPA" ]; then
        kubectl delete hpa ${hpa} -n kube-system
    fi
done

stop() {
    tag=$1
    echo Stopping $tag processes: 

    for pid in `ps aux | grep $tag | awk '{print $2}'`; do 
        echo killing pid = $pid
        sudo kill -9 $pid
    done
    echo Stopped $tag processes
}


stop monitor
stop workload
stop autoscaler
sshpass -p "Ubuntu1!" ssh ubuntu@ksurf-scaler "echo 'Ubuntu1!' | sh stop_tests.sh"
