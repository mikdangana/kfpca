from datetime import datetime
from jproperties import Properties
from kafka import KafkaProducer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta
import numpy as np
import time


class PoissonProducer:

    self.config = None
    self.num_points = 0.0
    # Topic name
    self.topic = ""
    self.producer = None


    def __init__(self):
        self.config = self.load_configs()
        self.num_points = float(self.config.get("poisson.n_per_t"))
        self.topic = config.get("kafka.topics").split(",")[0]
        # Bootstrap servers - you can input multiple ones with comma seperated.
        # for example, bs1:9092, bs2:9092
        bootstrap_servers = config.get("kafka.endpoints").split(",")
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers[0])


    @staticmethod
    def load_configs(self, path=os.path.join(os.path.dirname(__file__), 
                                             '../config/testbed.properties')):
        config = Properties()
        with open(path, 'rb') as config_file:
            config.load(config_file)
        return config


    def poisson_next_ms(self, t):
        plambda = float(config.get("poisson.lambda"))
        n = np.range(0, self.num_points)
        prob = np.power(plambda*t,n)/np.math.factorial(n)*np.exp(-plambda*t)
        return prob.argmax(axis=1)


    def poisson_send(self, msg_bytes):
        ts = self.poisson_next_ms(t)/self.num_points
        print(f'Sending message after {ts} seconds')
        time.sleep(ts)
        self.producer.send(self.topic, msg_bytes)
        result = future.get(timeout=60)
        self.producer.flash()


poisson = PoissonProducer()

for t in range(100):
    ts = datetime.now() - timedelta(minutes=60)
    # Convert to epoch milliseconds
    ts_ms = ts.timestamp()*1000.0
    poisson.poisson_send(bytes(f'{ts_ms}', 'utf-8'))
