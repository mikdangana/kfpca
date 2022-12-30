from datetime import datetime
from kafka import KafkaConsumer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta


# Topic name
topic = config.get("kafka.topics").split(",")[0]
# Bootstrap servers - you can input multiple ones with comma seperated.
# for example, bs1:9092, bs2:9092
bootstrap_servers = config.get("kafka.endpoints").split(",")
consumer = KafkaConsumer(
    topic, bootstrap_servers=bootstrap_servers)

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

print(offsets)

offset_p0 = offsets[topic_partition_0]

print(offset_p0)

consumer.seek(partition=topic_partition_0, offset=offset_p0.offset)

while True:
    print('polling...')
    records = consumer.poll(timeout_ms=1000)
    for topic_data, consumer_records in records.items():
        for consumer_record in consumer_records:
            print(str(consumer_record.value.decode('utf-8')))
        continue

