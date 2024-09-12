# Relative_keys

This is the code repository for paper "Relative Keys: Putting Feature Explanation into Context".

It primarily includes data preprocessing and model training (1), testing the explanations and monitoring explanations of SRK (2.1), OSRK (2.2), and SSRK (2.3) algorithms. Additionally, it involves testing explanation performance under dynamic models (2.4) and acts as an indicator for monitoring ML performance (2.5). It also encompasses a specific task of testing entity linking (3). Lastly, it includes a simple interface script for receiving instances from Redis (4). In summary, the core code includes (1), 2.(1) to 2.(3).

Firstly, the following packages are necessary:
```
numpy 1.20.3
pandas 2.0.1
scikit-learn 0.24.2
xgboost 1.7.1
redis 4.6.0
```

To test entity linking, the 'certa' package must also be installed to train the Ditto model. Please refer to https://github.com/tteofili/certa

We should configure a config file (the meanings of specific parameters have been clearly defined). The default file `config.yaml`uses the revidivism dataset as an example. More datasets please refer to `data_process` folder.

### 1 Train xgboost and get other necessary information.

```
python preprocess.py
```

With the trained model and the inference set, we can test all the algorithms. Make sure the corresponding folder exists.

### 2.1 test srk

To test SRK, run below script:

```
python main_srk.py
```

The average results will be printed on the command console, and the specific explanations for each instance will be stored in the `results` folder.

### 2.2 test osrk

To test OSRK, run below script:

```
python main_osrk.py
```

The average results will be printed on the command console, and the specific explanations for each instance will be stored in the `results` folder.

### 2.3 test ssrk

To test SSRK, run below script:

```
python main_ssrk.py
```

The average results will be printed on the command console, and the specific explanations for each instance will be stored in the `results` folder.


### 2.4 test dynamic performance

To evaluate the capability in explaining dynamic models that change over time during model inference, run below script:

```
python main_dynamic_nosignal.py
```

The average results will be printed on the command console.

### 2.5 test the effectiveness of monitoring ML performance

As an application of relative key monitoring, OSRK can be used to monitor the performance (accuracy) of blackbox ML during model serving. 

We must set noise_flag=True in the `config.yaml`.

```
python main_indicator.py
```

### 3 test entity matching

To generate and evaluate the keys for entity matching task, run

```
python test_er.py
```

### 4 redis interface
We have also developed a very simple interface `redis_inter.py` to redis to receive data from redis. 
Make sure the redis server is turned on.
