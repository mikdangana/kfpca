# Tuning Kafka using KF-PCA tracking algorithm

## What is KF-PCA
Papers on [Ksurf](https://ieeexplore.ieee.org/abstract/document/10776180) & [Ksurf+](https://doi.org/10.36227/techrxiv.174319529.94024645/v1)

## Disclaimer
This readme is a work in progress and may not yet cover all the features outlined in the papers above

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

## How to run the Kafka estimator
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


### Run Adversarial Tests
```
python src/adversarial_tester.py -f data/analysis/data_kf_2_node_metrics.csv -x "ksurf-master CPU (m)"
```

or simply
```
python src/adversarial_tester.py
```
