# Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation
This is the author's official PyTorch implementation for ECBNN. This repo contains code for experiments in the **NeurIPS 2022** paper: 

[Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation]()

## Project Description
Human intelligence has shown remarkably lower latency and higher precision than most AI systems when processing non-stationary streaming data in real-time. Numerous neuroscience studies suggest that such abilities may be driven by **internal predictive modeling**. In this paper, we explore the possibility of introducing such a mechanism in unsupervised domain adaptation (UDA) for handling non-stationary streaming data for real-time streaming applications. We propose to **formulate internal predictive modeling as a continuous-time Bayesian filtering problem within the context of a stochastic dynamical system**. Such a dynamical system describes the dynamics of model parameters of a UDA model evolving with non-stationary streaming data. Building on such a dynamical system, we then develop extrapolative continuous-time Bayesian neural networks (ECBNN), which generalize existing Bayesian neural networks to represent temporal dynamics and allow us to extrapolate the distribution of model parameters before observing the incoming data, therefore effectively reducing the latency. Remarkably, our empirical results show that ECBNN is capable of continuously generating better distributions of model parameters along the time axis given historical data only, thereby achieving (1) training-free online adaptation with low latency, (2) gradually improved alignment between the source and target features and (3) gradually improved model performance over time during the real-time testing stage.

## Method Overview
<p align="center">
<img src="assets/ECBNN-model.png" alt="" data-canonical-src="assets/ECBNN-model.png" width="75%"/>
</p>

## How to run
### Experiment
NOTE: All our experiments are run on an AMD EPYC 7302P 16-core CPU and an RTX A5000 GPU.

Experiment settings:
* Python 3.8.13
* PyTorch 1.8.1
* CUDA 11.1

### Running experiments

#### Streaming Rotating MNIST $\rightarrow$ USPS
To train and evalute our ECBNN on source-testing and target-testing set: (To ensure the reproducibility, use 'CUBLAS_WORKSPACE_CONFIG=:4096:8' as prefix )
```
cd Streaming_Rotating_MNIST_USPS
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run.py --model_name ECBNN
```

To evaluate the trained model on OOD-testing set:

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluate.py --model_name ECBNN
```

#### Multi-view Lip Reading
* We followed the paper *End-to-End Multi-View Lipreading, S. Petridis, Y. Wang, Z. Li, M. Pantic. British Machine Vision Conference. London, September 2017* to download and process the dataset. For your convenience, we also upload our [processed OuluVS2 dataset](). You can put the data as follows:

```
OuluVS2
`-- data
    `-- unipadding
        |-- train.pt
        |-- valid.pt
        |-- test.pt
```

* To train our ECBNN model from scratch (To ensure the reproducibility, use 'CUBLAS_WORKSPACE_CONFIG=:4096:8' as prefix)

```
cd OuluVS2
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run.py --model_name ECBNN
```

## Citation
If you use ECBNN or this codebase in your own work, please cite our paper:
```

```

## Acknowledgement
Part of the code is borrowed from [CIDA](https://github.com/hehaodele/CIDA). We thank the authors for releasing their codes.

## License
MIT license