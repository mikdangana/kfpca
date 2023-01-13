from datetime import datetime
from kafka import KafkaConsumer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta
from ../ekf import *
from ../utils import *
import os, re
import numpy as np
import subprocess as sp


class ActiveConsumer

    topic = None
    consumer = None
    config = None
    is_scaled = False
    kf = None

    def __init__(self):
        self.config = PoissonProducer.load_config()
        self.topic = self.config.get("kafka.topics").split(",")[0]
        bootstrap_servers = self.config.get("kafka.endpoints").split(",")
        self.consumer = KafkaConsumer(
            self.topic, bootstrap_servers=bootstrap_servers)
        self.kf = PCAKalmanFilter()


    def setup_listener(self):
        # Timestamp to look for events
        # Look for events created from 60 minutes ago
        ts = datetime.now() - timedelta(minutes=60)
        # Convert to epoch milliseconds
        ts_milliseconds = ts.timestamp()*1000.0

        print(f'Looking for offsets of : {ts} ({ts_milliseconds})')

        # We only retrieve from partition 0 for this example
        # as there is only one partition created for the topic
        # To find out all partitions, partitions_for_topic can be used.
        topic_partition_0 = TopicPartition(topic, 0)
        timestamps = {topic_partition_0: ts_milliseconds}
        offsets = consumer.offsets_for_times(timestamps)
        offset_p0 = offsets[topic_partition_0]
        print(f"offsets = {offsets}, offset_p0 = {offset_p0}")

        self.consumer.seek(partition=topic_partition_0, offset=offset_p0.offset)


    def get_latency_throughputs(self, records):
        metrics = {}
        ts = datetime.now() - timedelta(minutes=60)
        ts_ms = ts.timestamp()*1000.0
        for topic_data, consumer_records in records.items():
            for consumer_record in consumer_records:
                print(f"Consumer_record = {consumer_record}")
                k = f"{consumer_record.topic()}-{consumer_record.partition()}"
                latency = ts_ms - float(consumer_record.value.decode("utf-8"))
                metrics[k] = np.array([1, latency]) if k not in metrics else \
                             np.array([1, latency]) + metrics[k]
        def means([size, total_latency]):
            mean_latency = total_latency / size
            throughput_recs = size / mean_latency
            print(f'mean_latency = {latency}, throughput = {throughput_recs}')
            return mean_latency, throughput_recs
        tuples = map(lambda i: i[0].split("-") + means(i[1]), metrics.items())
        return list(tuples)

    
    def means(self, [size, total_latency]):
        mean_latency = total_latency / size
        throughput_recs = size / mean_latency
        print(f'mean_latency = {latency}, throughput = {throughput_recs}')
        return mean_latency, throughput_recs


    def to_metrics(self, requests, metrics, track_cadence):
        for i, consumer_record in zip(range(len(requests)), requests):
            val = float(consumer_record.value.decode("utf-8"))
            msmts.append(val)
            if i % track_cadence < 1 and len(msmts) and len(msmts[0]) > 1:
                # compute jacobian
                latencies.append(ts_ms - val)
                kf.update(msmts[-1], Hj=to_jacobian(msmts[-2:], latencies[-2:]))
            else:
                # track latency/throughput
                latencies.append(kf.predict(msmts[-1]))
            print(f"Consumer_record = {consumer_record}")
            k = f"{consumer_record.topic()}-{consumer_record.partition()}"
            metrics[k] = np.array([1, latencies[-1]]) if k not in metrics else \
                         np.array([1, latencies[-1]]) + metrics[k]
            return metrics


    def track_latency_throughputs(self, records):
        metrics, msmts, latencies = {}, [], []
        ts = datetime.now() - timedelta(minutes=60)
        ts_ms = ts.timestamp()*1000.0
        kf_update_rate = self.config.get("kf.update.rate")
        mod = 0 if kf_update_rate<=0 else 1/kf_update_rate
        for topic_data, requests in records.items():
            #requests.sort(key=lambda rec : float(rec.value.decode("utf-8")))
            self.to_metrics(requests, metrics, mod)
        tuples = map(lambda i: i[0].split("-") + means(i[1]), metrics.items())
        pickleconc(self.config.get("data.tracking.file"), tuples)
        return list(tuples)
  

    def get_leader(self, topic, partition):
        topics_file = "topics.txt"
        kafkadir = self.config.get("kafka.home")
        os.system(f"{kafkadir}/bin/kafka-topics.sh --zookeeper localhost:2181" \
                  " --describe --topic {topic} > {topics_file}")
        ldr, lines = 0, open(topics_file, 'r').readlines()
        for line in lines:
            if line.contains('Partition: ' + partition + ' '):
                ldr = re.compile(r' Leader: ([0-9,]+)').match(line).groups()[0]

        return ldr


    def scale_broker(self, topic, partition, scale_up=True):
        ldr = self.get_leader(topic, partition)     
        
        kafka_home = self.config.get("kafka.home")
        sleep_ts = self.config.get("kafka.startup_time_sec")
        cfg_file = f'{kafka_home}_broker{ldr}/config/server.properties'
        q_sz = float(PoissonProducer.load_config(cfg_file)
                                    .get("queued.max.requests"))
        q_sz += 1 if scale_up else -1
        os.system(f"sed -i 's/(queued.max.requests*=).*/\1{q_sz}/' {cfg_file}")
        os.system(f"{kafka_home}_broker{ldr}/bin/kafka-server-stop.sh")
        time.sleep(sleep_ts)
        os.system(f"{kafka_home}_broker{ldr}/bin/kafka-server-start.sh" \
                   " {kafka_home}_broker{ldr}/config/server.properties")
        time.sleep(sleep_ts)
        self.is_scaled = scale_up
        return self.is_scaled


    def listen(self):
        self.setup_listener()
        self.max_latency = float(self.config.get("max_latency"))
        self.min_throughput = float(self.config.get("min_throughput"))
        while True:
            print('polling...')
            records = self.consumer.poll(timeout_ms=1000)
            for topic, partition, latency, throughput in \
                    get_latency_throughputs(records):
              print("topic={topic}, partition={partition}, " \
                    "latency={latency}, throughput={throughput}") 
              if latency > max_latency or throughput < min_throughput:
                if !self.is_scaled:
                    # scale up broker queue size
                    scale_broker(topic, partition)
              else if self.is_scaled:
                # scale down broker queue size
                scale_broker(topic, partition, False)


if __name__ == "__main__":
    consumer = ActiveConsumer()
    consumer.listen()
