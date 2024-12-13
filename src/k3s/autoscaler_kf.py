import time
import subprocess
import json
from ekf import ExtendedKalmanFilter, PCAKalmanFilter, predict

# Configuration
NAMESPACE = "kube-system"
DEPLOYMENT_NAME = "kube-app"
CPU_THRESHOLD = 80   # Percentage
MEMORY_THRESHOLD = 80  # Percentage
CHECK_INTERVAL = 30  # Seconds
KF_TUNE_CSV = "kf_tune.csv"
KF_TUNE_COL = "cpu"


msmts = []
ekf = None


def get_metrics():
    try:
        output = subprocess.check_output(
            ["kubectl", "top", "pods", "-n", NAMESPACE, "-o", "json"],
            stderr=subprocess.STDOUT
        )
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching metrics: {e.output.decode()}")
        return None

def apply_attention_filter(current_value, observed_value):
    global msmts, ekf
    #ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    #ekf.x = current_value  # Initial state
    #ekf.P *= 1000         # Initial uncertainty
    #ekf.R = 5             # Measurement uncertainty
    #ekf.Q = 0.1           # Process uncertainty
    #ekf.F = 1             # State transition matrix
    #ekf.H = 1             # Measurement function

    #ekf.predict()
    #ekf.update(observed_value)

    #return ekf.x[0]
    if ekf is None:
        ekf = PCAKalmanFilter(nmsmt=2, dx=2, normlalize=True, 
                              att_fname=KF_TUNE_CSV, att_col=KF_TUNE_COL)
    msmts.append(observed_value)
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
            ["kubectl", "scale", "deployment", DEPLOYMENT_NAME, f"--replicas={replicas}", "-n", NAMESPACE]
        )
        print(f"Scaled deployment to {replicas} replicas.")
    except subprocess.CalledProcessError as e:
        print(f"Error scaling deployment: {e.output.decode()}")

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

