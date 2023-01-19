from datetime import datetime
from jproperties import Properties
from kafka import KafkaProducer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta
from scipy.special import factorial
import numpy as np
import os, time


class PoissonProducer:

    config = None
    num_points = 0.0
    # Topic name
    topic = ""
    producer = None


    def __init__(self):
        self.config = self.load_configs()
        self.num_points = float(self.config.get("poisson.n_per_t").data)
        self.topic = self.config.get("kafka.topics").data.split(",")[0]
        # Bootstrap servers - you can input multiple ones with comma seperated.
        # for example, bs1:9092, bs2:9092
        bootstrap_servers = self.config.get("kafka.endpoints").data.split(",")
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers[0])


    @staticmethod
    def load_configs(path=os.path.join(os.path.dirname(__file__), '..',
                                       '..', 'config', 'testbed.properties')):
        config = Properties()
        with open(path, 'rb') as config_file:
            config.load(config_file)
        return config


    def poisson_next_ms(self, t):
        plambda = float(self.config.get("poisson.lambda").data)
        n = np.arange(0, self.num_points)
        prob = np.power(plambda*t,n)/factorial(n)*np.exp(-plambda*t)
        #print(f"prob.argmax = {prob.argmax(axis=0)}")
        return prob.argmax(axis=0)


    def poisson_send(self, t, msg_bytes):
        ts = self.poisson_next_ms(t)/self.num_points
        print(f'Sending message after {ts} seconds')
        time.sleep(ts)
        future = self.producer.send(self.topic, msg_bytes)
        result = future.get(timeout=60)
        self.producer.flush()



if __name__ == "__main__":

    poisson = PoissonProducer()

    for t in range(100):
        ts = datetime.now() - timedelta(minutes=60)
        # Convert to epoch milliseconds
        ts_ms = ts.timestamp()*1000.0
        poisson.poisson_send(t, bytes(f'{ts_ms}', 'utf-8'))

