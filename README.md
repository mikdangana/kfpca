# Tuning Kafka using KF-PCA tracking algorithm

## Setup
```
vim config/testbed.properties
```
Set Tracker properties

### If not installed
```
pip install -r requirements.txt
```

## Get most recent data from Twitter API
```
python src/twitter_search.py
```

## How to run the Kafka tracker
```
sh src/kafka/setup.sh -h
python -m flask run
```
OR
```
python src/kafka/consumer.py
```

### Single test
```
sh src/kafka/setup.sh  --runtest
```

### Multi-test
```
sh src/kafka/setup.sh --runtests 10
```


## Result data
```
ls data_tracker.pickle.csv*
```


## Workload data
```
ls data/
```

## KF accuracy tests
### Edit tracker.type = KF/UKF/EKF/EKF-PCA/AKF-PCA config
```
sed -i s/tracker.type=.*/tracker.type=EKF-PCA/ config/testbed.properties
```

### Run KF csv-based estimator test
```
python src/ekf.py --testpcacsv -f data/twitter_trace.csv -x 'Tweet Count' -y 'Tweet Count' > out_ekfpca.txt
```

### View summary stats 
```
tail out_ekfpca.txt
```
