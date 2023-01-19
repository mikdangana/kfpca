from flask import Flask, request
from producer import *
import logging, sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
from ekf import *
from utils import *

app = Flask(__name__)


class Tracker:

    config = None
    is_scaled = False
    kf = None

    def __init__(self):
        self.config = PoissonProducer.load_configs()
        self.kf = PCAKalmanFilter()


    def means(self, metrics):
        size, total_latency = metrics
        mean_latency = total_latency / size
        throughput_recs = size / mean_latency
        print(f'mean_latency = {latency}, throughput = {throughput_recs}')
        return mean_latency, throughput_recs


    def to_jacobian(y, x):
        dy = np.array(y) - np.array(y[0:-1])
        dx = np.array(x) - np.array(x[0:-1])
        J = np.multiply(np.array(list(map(lambda t: [t], dy))), np.array(dx))
        return lambda x : np.multiply(J, np.array(x))


    def to_metrics(self, requests, metrics, track_cadence):
        msmts = []
        for i, consumer_record in zip(range(len(requests)), requests):
            val = float(consumer_record.value.decode("utf-8"))
            msmts.append(val)
            if i % track_cadence < 1 and len(msmts) and len(msmts[0]) > 1:
                # compute jacobian
                latencies.append(ts_ms - val)
                kf.update(msmts[-1], Hj=to_jacobian(msmts[-2:], latencies[-2:]))
            else:
                # track latency/throughput
                latencies.append(kf.predict(msmts[-2:])[-1])
            print(f"Consumer_record = {consumer_record}")
            k = f"{consumer_record.topic()}-{consumer_record.partition()}"
            metrics[k] = np.array([1, latencies[-1]]) if k not in metrics else \
                         np.array([1, latencies[-1]]) + metrics[k]
            return metrics


    def track_latency_throughputs(self, records):
        metrics, msmts, latencies = {}, [], []
        ts = datetime.now() - timedelta(minutes=60)
        ts_ms = ts.timestamp()*1000.0
        update_rate = float(self.config.get("tracker.update.rate").data)
        mod = 0 if update_rate<=0 else 1/update_rate
        for topic_data, requests in records.items():
            #requests.sort(key=lambda rec : float(rec.value.decode("utf-8")))
            self.to_metrics(requests, metrics, mod)
        tuples = map(lambda i: i[0].split("-") + means(i[1]), metrics.items())
        pickleconc(self.config.get("data.tracker.file").data, list(tuples))
        print("Tracker output in {}".format(
              self.config.get("data.tracker.file").data))
        return list(tuples)
  

    def get_leader(self, topic, partition):
        topics_file = "topics.txt"
        kafkadir = self.config.get("kafka.home").data
        os.system(f"{kafkadir}/bin/kafka-topics.sh --zookeeper localhost:2181" \
                  " --describe --topic {topic} > {topics_file}")
        ldr, lines = 0, open(topics_file, 'r').readlines()
        for line in lines:
            if line.contains('Partition: ' + partition + ' '):
                ldr = re.compile(r' Leader: ([0-9,]+)').match(line).groups()[0]

        return ldr


    def scale_broker(self, topic, partition, scale_up=True):
        ldr = self.get_leader(topic, partition)     
        
        kafka_home = self.config.get("kafka.home").data
        sleep_ts = float(self.config.get("kafka.startup_time_sec").data)
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



    def process(self, records):
          for topic, partition, latency, throughput in \
                    self.track_latency_throughputs(records):
              print("topic={topic}, partition={partition}, " \
                    "latency={latency}, throughput={throughput}") 
              if latency > max_latency or throughput < min_throughput:
                if not self.is_scaled:
                    # scale up broker queue size
                    self.scale_broker(topic, partition)
              elif self.is_scaled:
                # scale down broker queue size
                self.scale_broker(topic, partition, False)


tracker = Tracker()

@app.route("/")
def home():
    app.logger.info("Home visited")
    logging.info("home")
    return "<p>Hyscale KFPCA-based Kafka Tracker</p>"


@app.route("/track", methods=['POST'])
def track_requests():
    app.logger.info("Track requests = %s", request.data)
    tracker.process(request.form('records'))
