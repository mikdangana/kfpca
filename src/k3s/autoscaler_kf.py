import time
import json
import numpy as np
from kubernetes import client, config
import sys, tensorflow as tf
sys.path.append('/root/kfpca/src')
from ekf import ExtendedKalmanFilter, PCAKalmanFilter, predict

# Configuration
NAMESPACE = "kube-system"
DEPLOYMENT_NAME = sys.argv[sys.argv.index('-d')+1] if '-d' in sys.argv else "kube-app" 
CPU_THRESHOLD = 5   # Percentage
MEMORY_THRESHOLD = 800  # MiB
CHECK_INTERVAL = 30  # Seconds
KF_TUNE_CSV = "kf_tune_converted.csv"
KF_TUNE_COL = "ksurf-worker2-27aec434 CPU (m)"
KUBECONFIG = "/home/ubuntu/.kube/config"

msmts = np.array([0])
ekf = None

print("Available devices:", tf.config.list_physical_devices())

def get_metrics():
    try:
        # Load Kubernetes configuration
        config.load_kube_config(config_file=KUBECONFIG)
        metrics_api = client.CustomObjectsApi()

        # Query pod metrics
        metrics = metrics_api.list_namespaced_custom_object(
            group="metrics.k8s.io", 
            version="v1beta1", 
            namespace=NAMESPACE, 
            plural="pods"
        )

        deployment_pods = [
            pod for pod in metrics['items'] 
            if pod['metadata']['labels'].get('app') == DEPLOYMENT_NAME
        ]

        return {"items": [
            {
                "containers": [
                    {
                        "usage": {
                            "cpu": pod['containers'][0]['usage']['cpu'].replace('n', ''),
                            "memory": pod['containers'][0]['usage']['memory'].replace('Mi', '')
                        }
                    }
                ]
            }
            for pod in deployment_pods
        ]}

    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None


def apply_attention_filter(observed_value):
    global msmts, ekf
    if ekf is None:
        ekf = PCAKalmanFilter(nmsmt=2, dx=2, normalize=True, att_fname=KF_TUNE_CSV, att_col=KF_TUNE_COL)
    msmts = np.append(msmts, [observed_value])
    return predict(ekf, msmts, dy=2)


def check_thresholds(metrics):
    avg_cpu = 0
    avg_memory = 0
    pod_count = len(metrics['items']) if len(metrics['items']) else 1

    for pod in metrics['items']:
        cpu_usage = int(pod['containers'][0]['usage']['cpu'])
        memory_usage = int(pod['containers'][0]['usage']['memory'])

        avg_cpu += apply_attention_filter(cpu_usage / 10)
        avg_memory += apply_attention_filter(memory_usage)

    avg_cpu, avg_memory = avg_cpu / pod_count, avg_memory / pod_count
    cpu_exceeded = avg_cpu >= CPU_THRESHOLD
    memory_exceeded = avg_memory >= MEMORY_THRESHOLD
    print(f"cpu = {avg_cpu}, threshold = {CPU_THRESHOLD}")

    return cpu_exceeded or memory_exceeded


def scale_deployment(replicas):
    try:
        subprocess.check_call(
            ["kubectl", "scale", "deployment", DEPLOYMENT_NAME, f"--replicas={replicas}", "-n", NAMESPACE, f"--kubeconfig={KUBECONFIG}"]
        )
        print(f"Scaled deployment to {replicas} replicas.")
    except subprocess.CalledProcessError as e:
        print(f"Error scaling deployment: {e.output}")


def main():
    current_replicas = 1
    while True:
        metrics = get_metrics()
        if metrics and check_thresholds(metrics):
            current_replicas += 1
            scale_deployment(current_replicas)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

