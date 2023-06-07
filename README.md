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


## Results
```
ls data/
```
