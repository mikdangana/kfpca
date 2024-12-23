import time
import subprocess
import json, numpy as np
import sys, tensorflow as tf
sys.path.append('/root/kfpca/src')
print(f"sys.path = {sys.path}")
from ekf import ExtendedKalmanFilter, PCAKalmanFilter, predict

# Configuration
NAMESPACE = "kube-system"
DEPLOYMENT_NAME = sys.argv[sys.argv.index('-d')+1] \
                  if '-d' in sys.argv else "kube-app" 
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
        output = subprocess.check_output(
            ["kubectl", "top", "pods", "-n", NAMESPACE, "-l", f"app={DEPLOYMENT_NAME}", f"--kubeconfig={KUBECONFIG}"],
            stderr=subprocess.STDOUT
        )
        lines = output.decode().splitlines()
        metrics = {"items": []}
        if len(lines) > 1:
            headers = lines[0].split()
            for line in lines[1:]:
                values = line.split()
                pod_data = dict(zip(headers, values))
                metrics["items"].append({
                    "containers": [{
                        "usage": {
                            "cpu": pod_data["CPU(cores)"].replace('m', ''),
                            "memory": pod_data["MEMORY(bytes)"].replace('Mi', '')
                        }
                    }]
                })
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error fetching metrics: {e.output.decode()}")
        print(["kubectl", "top", "pods", "-n", NAMESPACE, f"--kubeconfig={KUBECONFIG}"])
        return None


def apply_attention_filter(current_value, observed_value):
    global msmts, ekf
    if ekf is None:
        ekf = PCAKalmanFilter(nmsmt=2, dx=2, normalize=True, 
                              att_fname=KF_TUNE_CSV, att_col=KF_TUNE_COL)
    msmts = np.append(msmts, [observed_value])
    return predict(ekf, msmts, dy=2)


def check_thresholds(metrics):
    avg_cpu = 0
    avg_memory = 0
    pod_count = len(metrics['items'])

    for pod in metrics['items']:
        cpu_usage = int(pod['containers'][0]['usage']['cpu'].replace('m', ''))
        memory_usage = int(pod['containers'][0]['usage']['memory'].replace('Mi', ''))

        avg_cpu = apply_attention_filter(avg_cpu, cpu_usage / 10)
        avg_memory = apply_attention_filter(avg_memory, memory_usage)

    cpu_exceeded = avg_cpu >= CPU_THRESHOLD
    memory_exceeded = avg_memory >= MEMORY_THRESHOLD

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

