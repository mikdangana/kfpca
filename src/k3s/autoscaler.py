import time
import subprocess
import json

# Configuration
NAMESPACE = "kube-system"
DEPLOYMENT_NAME = "kube-app"
CPU_THRESHOLD = 80   # Percentage
MEMORY_THRESHOLD = 80  # Percentage
CHECK_INTERVAL = 30  # Seconds

def get_metrics():
    try:
        output = subprocess.check_output(
            ["kubectl", "top", "pods", "-n", NAMESPACE],
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
        return None


def check_thresholds(metrics):
    cpu_exceeded = False
    memory_exceeded = False

    for pod in metrics['items']:
        cpu_usage = int(pod['containers'][0]['usage']['cpu'].replace('m', ''))
        memory_usage = int(pod['containers'][0]['usage']['memory'].replace('Mi', ''))
        
        # Assuming CPU is in millicores and Memory in MiB
        if cpu_usage / 10 >= CPU_THRESHOLD:
            cpu_exceeded = True

        if memory_usage >= MEMORY_THRESHOLD:
            memory_exceeded = True

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

