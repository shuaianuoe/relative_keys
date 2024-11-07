# Relative_keys

This is the code repository for paper "Relative Keys: Putting Feature Explanation into Context".

It primarily includes data preprocessing and model training (1), testing the explanations and monitoring explanations of SRK (2.1), OSRK (2.2), and SSRK (2.3) algorithms. Additionally, it involves testing explanation performance under dynamic models (2.4) and acts as an indicator for monitoring ML performance (2.5). It also encompasses a specific task of testing entity linking (3). In summary, the core code includes (1), 2.(1) to 2.(3).

Firstly, the necessary packages are specified in `requirements.txt`.  Run below code:
```
pip install -r requirements.txt
```

We should configure a config file (the meanings of specific parameters have been clearly defined). The default file `config.yaml`uses the revidivism dataset as an example. More datasets please refer to `data_process` folder.

All the scripts should run on Windows 10 as well as on Ubuntu 14.04 and later systems.

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

If you want to test the entity linking task, you need to replace the datasetsname in `config.yaml` with an entity linking dataset.

Using DBLP-ACM as an example:

1. Change the datasetsname in config.yaml to "DBLP-ACM".

2. To train the entity linking model and generate the instance to be explained, run the command in the `certamain` folder:
```
python train_certa.py
```

3. To generate and evaluate the keys for entity matching task, run the command in the `certamain` folder:
```
python test_er.py
```

NOTE: (1) Step2 can be time-consuming. For DBLP-ACM, it takes approximately 30 minutes. (2) If you encounter the 'KeyError: 'certa'' error, simply restart the console.


