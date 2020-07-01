# Attentive Traffic Flow Machines


This is a PyTorch implementation of **Attentive Traffic Flow Machines (ATFM)**. ATFM is a a
unified neural network which can effectively learn the spatial-temporal feature representations of crowd flow with an attention mechanism.

If you use this code for your research, please cite our papers （[Conference Version](https://dl.acm.org/citation.cfm?id=3240681) and [Journal Version](https://ieeexplore.ieee.org/document/9127874)):

```
@inproceedings{liu2018attentive,
  title={Attentive Crowd Flow Machines},
  author={Liu, Lingbo and Zhang, Ruimao and Peng, Jiefeng and Li, Guanbin and Du, Bowen and Lin, Liang},
  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
  pages={1553--1561},
  year={2018},
  organization={ACM}
}
```

```
@article{liu20120dynamic,
  title={Dynamic Spatial-Temporal Representation Learning for Traffic Flow Prediction},
  author={Liu, Lingbo and Zhen, Jiajie and Li, Guanbin and Zhan, Geng and He, Zhaocheng and Du, Bowen and Lin, Liang},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2020}
}
```

## Requirements
- torch==0.4.1

## Preprocessing
**For Crowd Flow Prediction:**  download [TaxiBJ](https://github.com/lucktroy/DeepST/tree/master/data/TaxiBJ/) / [BikeNYC](https://github.com/lucktroy/DeepST/tree/master/data/BikeNYC) and put them into folder  ```data/TaxiBJ``` and ```data/BikeNYC```.

**For Citywide Passenger Demand Prediction (CPDP):**  the dataset of CPDP has been in folder  ```data/TaxiNYC```.

## Model Training
```bash
# TaxiBJ
python run_taxibj.py

# BikeNYC
python run_bikenyc.py

# TaxiNYC
python run_taxinyc.py
```

## Testing
```bash
# TaxiBJ
python test_taxibj.py

# BikeNYC
python test_bikenyc.py

# TaxiNYC
python test_taxinyc.py
```


