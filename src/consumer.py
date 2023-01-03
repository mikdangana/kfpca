from datetime import datetime
from kafka import KafkaConsumer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta
import os, re
import subprocess as sp


class ActiveConsumer

    topic = None
    consumer = None
    config = None
    is_scaled = False

    def __init__(self):
        self.config = PoissonProducer.load_config()
        self.topic = self.config.get("kafka.topics").split(",")[0]
        bootstrap_servers = self.config.get("kafka.endpoints").split(",")
        self.consumer = KafkaConsumer(
            self.topic, bootstrap_servers=bootstrap_servers)


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


    def get_latency_throughput(self, records):
        ts = datetime.now() - timedelta(minutes=60)
        # Convert to epoch milliseconds
        ts_ms = ts.timestamp()*1000.0
        total_latency = 0
        for topic_data, consumer_records in records.items():
            for consumer_record in consumer_records:
                print(str(consumer_record.value.decode('utf-8')))
                print(f"Consumer_record = {consumer_record}")
                latency = ts_ms - float(consumer_record.value.decode("utf-8"))
                total_latency += latency
        mean_latency = total_latency / records.items().size()
        throughput_recs = records.items.size() / mean_latency
        print(f'mean_latency = {latency}, throughput = {throughput_recs}')
        return mean_latency, throughput_recs
  

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
        while True:
            print('polling...')
            records = self.consumer.poll(timeout_ms=1000)
            latency, throughput = get_latency_throughput(records)
            self.max_latency = float(self.config.get("max_latency"))
            self.min_throughput = float(self.config.get("min_throughput"))
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
