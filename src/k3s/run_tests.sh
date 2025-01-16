
TEST_TYPE="HPA" # HPA|KF|SIMPLE

for w in $(seq 1 1 20); do
    hpa=kafka-worker${w}-hpa
    host=ksurf-worker${w}
    kubectl delete hpa ${hpa} -n kube-system
    suffix=""
    if [ $TEST_TYPE = "KF" ]; then suffix=_kf; fi
    if [ $TEST_TYPE = "HPA" ]; then
        ssh ubuntu@$host "echo 'Ubuntu1!' | sudo kubectl apply -f /root/hpa.yaml --kubeconfig=/home/ubuntu/.kube/config"
	kubectl describe hpa ${hpa} -n kube-system
    else
        #nohup sudo /root/kafka-env/bin/python3 /root/kfpca/src/k3s/autoscaler${suffix}.py -d kafka-worker${w} > /home/ubuntu/logs/autoscaler${w}.log 2>&1 &
        nohup sshpass -p "Ubuntu1!" ssh ubuntu@ksurf-scaler "echo 'Ubuntu1!' | sudo /root/kafka-env/bin/python3 /root/kfpca/src/k3s/autoscaler${suffix}.py -d kafka-worker${w} > /home/ubuntu/autoscaler${w}.log 2>&1" > auto${w}.log 2>&1 &
    fi
done

if [ ! -e logs ]; then mkdir /home/ubuntu/logs; fi

for p in $(seq 1 1 5); do 
    sshpass -p "Ubuntu1!" ssh ubuntu@ksurf-producer${p} "echo 'Ubuntu1!' | sudo nohup sh /root/kafka_producer_workload.sh > /home/ubuntu/logs/workload.log 2>&1 &"
done

sh run_monitors.sh

sh stop_tests.sh

