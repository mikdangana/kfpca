from flask import Flask, request
from producer import *
from time import sleep
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
        self.kf = PCAKalmanFilter(nmsmt = 2, dx = 2)
        pickledump(self.config.get("data.tracker.file").data, 
                   [["TOPIC", "PARTITION", "MEAN LATENCY", "MEAN THROUGHPUT"]])


    def means(self, metrics):
        size, total_latency = metrics
        mean_latency = total_latency / size
        throughput_recs = size / mean_latency
        print(f'mean_latency = {mean_latency}, throughput = {throughput_recs}')
        return [mean_latency, throughput_recs]


    def to_jacobian(self, y, x):
        dy = np.subtract(np.array([y[-1],y[-1]]), np.array([y[-2], y[-2]]))
        dx = np.subtract(np.array(x[-1]), np.array(x[-2]))
        J = np.array([[dy[r]/dx[c] for c in [0,1]] for r in [0,1]])
        print(f"to_jacobian.y={y}, x={x}, dy={dy}, J={J}")
        return J


    def Hj(self, y, x):
        J = self.to_jacobian(y, x)
        return lambda x: J

    
    def Hx(self, y, x):
        J = self.to_jacobian(y, x)
        def H(x1):
            x1 = x1.T[0]
            dx = np.subtract(np.array(x1), np.array(x[-1]))
            Jx = np.matmul(J, np.subtract(np.array(x1), np.array(x[-1])))
            Hx = np.add(np.array(y[-2:]), Jx)
            print(f"H().x={x}, x1={x1}, Jx={Jx}, dx={dx}, Hx = {Hx}")
            return Hx
        return H


    def pair(self, items, cadence, defval=0):
        return [items[-1-int(cadence)] if len(items)>cadence else defval,
                items[-1]]


    def to_metrics(self, requests, metrics, track_cadence):
        msmts, latencies, n, hj,hx = [0,0], [[0, 0]], len(requests), None, None
        ts_ms, t0 = datetime.now().timestamp()*1000.0, None
        print(f"to_metrics.requests = {requests}, len = {len(requests)}")
        for i, consumer_record in zip(range(n), requests):
            val = float(consumer_record.value.decode("utf-8"))
            t0 = val if t0 is None else t0
            msmts.append(self.kf.normalize(val - t0))
            if i % track_cadence < 1 or len(msmts) == 3:
                # compute jacobian, Hx, and normalized throughput 
                latencies.append([ts_ms - val, (n-i)*1000*100000/(ts_ms-val)])
                self.kf.ekf.x = np.array([[v] for v in latencies[-1]])
                x = self.pair(latencies, track_cadence, [0,0])
                y = self.pair(msmts, track_cadence)
                print(f"to_metrics.ekf.x0 = {self.kf.ekf.x}, y={y}, x={x}")
                hj, hx = self.Hj(y, x), self.Hx(y, x)
                self.kf.update(msmts[-2:], Hj=hj, H=hx)
            else:
                print(f"msmts[-2:] = {msmts[-2:]}, pred = {self.kf.ekf.x}, " \
                      f"prior={self.kf.ekf.x_prior}, latencies={latencies[-1]}")
                # track latency/throughput
                latencies.append(list(self.kf.predict([msmts[-1],msmts[-1]], 
                                       Hj=hj, H=hx)[-1][-1].T[0]))
            print(f"to_metricks.ekf.x_prior = {self.kf.ekf.x_prior}")
            k = f"{consumer_record.topic()}-{consumer_record.partition()}"
            metrics[k] = [latencies[-1]] if k not in metrics else \
                         metrics[k] + [latencies[-1]]
        return metrics


    def track_latency_throughputs(self, records):
        metrics = {}
        update_rate = float(self.config.get("tracker.update.rate").data)
        mod = 0 if update_rate<=0 else 1/update_rate
        for topic_data, requests in records.items():
            print(f"track.topic = {topic_data}, requests = {len(requests)}")
            #requests.sort(key=lambda rec : float(rec.value.decode("utf-8")))
            self.to_metrics(requests, metrics, mod)
        rows = []
        for k, vals in metrics.items():
            #print(f"track_l_t.vals = {vals}")
            rows += [k.split("-") + v for v in vals]
        pickleconc(self.config.get("data.tracker.file").data, rows)
        print("Tracker output in {}".format(
              self.config.get("data.tracker.file").data))
        return rows
  

    def get_leader(self, topic, partition):
        topics_file = "topics.txt"
        kafkadir = self.config.get("kafka.home").data
        os.system(f"{kafkadir}/bin/kafka-topics.sh --zookeeper localhost:2181" \
                  f" --describe --topic {topic} > {topics_file}")
        ldr, lines = 0, open(topics_file, 'r').readlines()
        for line in lines:
            if line.contains('Partition: ' + partition + ' '):
                ldr = re.compile(r' Leader: ([0-9,]+)').match(line).groups()[0]

        return ldr


    def scale_broker(self, topic, partition, scale_up=True):
        ldr = self.get_leader(topic, partition)     
        kafka_home = self.config.get("kafka.home").data
        sleep_ts = float(self.config.get("kafka.startup_time_sec").data)
        brokerhome = f'{kafka_home}_broker{ldr}'
        cfg_file = os.path.join(brokerhome, 'config', 'server.properties')
        cparam = "socket.receive.buffer.bytes"
                                
        q_sz = float(PoissonProducer.load_configs(cfg_file).get(cparam).data)
                                    
        q_sz += 1 if scale_up else (-1 if q_sz > 1 else 0)
        os.system(f"sed -i 's/\("+cparam+"*=\).*/\\1"+str(q_sz)+"/' "+cfg_file)
        os.system(os.path.join(brokerhome, "bin", "kafka-server-stop.sh"))
                               
        sleep(sleep_ts)
        os.system(os.path.join(brokerhome, "bin", "kafka-server-start.sh") + 
                  os.path.join(f" {brokerhome}", "config", "server.properties"))
                               
        sleep(sleep_ts)
        self.is_scaled = scale_up
        return self.is_scaled



    def process(self, records):
          max_latency = float(self.config.get("max_latency").data)
          min_throughput = float(self.config.get("min_throughput").data)
          for topic, partition, latency, throughput in \
                    self.track_latency_throughputs(records):
              print(f"topic={topic}, partition={partition}, " \
                    f"latency={latency}, throughput={throughput}") 
              if latency > max_latency or throughput < min_throughput:
                if not self.is_scaled:
                    # scale up broker queue size
                    #self.scale_broker(topic, partition)
                    print("scale_up")
              elif self.is_scaled:
                # scale down broker queue size
                #self.scale_broker(topic, partition, False)
                print("scale_down")


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


class ConsumerRecord:
    key = "k"
    value = "0".encode(encoding='utf-8')

    def __init__(self, key, value):
        self.key = key
        self.value = value.encode(encoding='utf-8')

    def topic(self):
        return "topic1"

    def partition(self):
        return "partition1"

        
if __name__ == "__main__":
    def get_ts(i):
        return (datetime.now() - timedelta(seconds=70-i)).timestamp()*1000.0
    print("tracker.process = {}".format(
        tracker.process(
            {"topic1": 
             [ConsumerRecord(f"k{i}", f"{get_ts(i)}") for i in range(10)]})))
                                      
