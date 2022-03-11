# Machine learning on knowledge graphs for context-aware security monitoring

This repository contains the dataset and model used in the publications [Machine learning on knowledge graphs for context-aware security monitoring (IEEE CSR 2021)](https://arxiv.org/abs/2105.08741) and [An energy-based model for neuro-symbolic reasoning on knowledge graphs (IEEE ICMLA 2021)](https://arxiv.org/abs/2110.01639).

## Introduction

Machine learning techniques are gaining attention in the context of intrusion detection due to the increasing amounts of data generated by monitoring tools, as well as the sophistication displayed by attackers in hiding their activity. However, existing methods often exhibit important limitations in terms of the quantity and relevance of the generated alerts. Recently, knowledge graphs are finding application in the cybersecurity domain, showing the potential to alleviate some of these drawbacks thanks to their ability to seamlessly integrate data from multiple domains using human-understandable vocabularies. Here, we propose a link prediction method for scoring anomalous activity in industrial systems. To evaluate our approach, a labelled dataset was generated from an industrial automation demonstrator. After initial unsupervised training, the proposed method is shown to produce intuitively well-calibrated and interpretable alerts in a diverse range of scenarios, hinting at the potential benefits of relational machine learning on knowledge graphs for intrusion detection purposes.

## Code

To install the package, use

`pip install -e .`

after which the package can be imported in Python using

`import enbed`

## Data

The data folder includes data generated from an industrial automation demonstrator that mimics the integration of OT and IT technologies.
Activity from this system is recorded and subsequently transferred into a knowledge graph.
The recorded data has been preprocessed for experiments (e.g., time stamps were dropped, variable accesses were solely modelled via edges, etc.), but alternative representations of the data are possible. 
For this reason, the original raw data is included as well.

The dataset is composed of a training set, which is a recording of baseline activity of the automation system, and a test set, where deviations from the baseline were added during test time according to standard cybersecurity attack patterns.
Test scenarios are separated into the following categories:

**ssh:** changes in ssh activity between system components. E.g., changes in connectivity patterns.

**https:** changes in https activity between system components or system components and the internet. E.g., changes in network traffic.

**credential_use:** applications start being hosted from other devices than usual.

**variables_access:** applications start reading or writing from variables of the OT system that were not accessed during the baseline.

**scan:** a device initiates a network scan.

Novel activity is assigned five different degrees of severity:

`0 - highly suspicious`

`1 - suspicious`

`2 - unexpected`

`3 - expected`

`4 - observed during training`

Data files contain subject, predicate and object (separated by tabs).
In case of the benchmarks, the files also include the label in a fourth column.
`.del` files contain the graph as ids, `.txt` files with string names.

## Experiments

Jupyter notebook where the energy-based graph embedding model is used to evaluate the severity of triple events in the individual benchmark cases.

## Citation

If you use the provided dataset or build upon the introduced approach to anomaly detection in your work, please cite

```
@inproceedings{garrido2021machine,
  title={Machine learning on knowledge graphs for context-aware security monitoring},
  author={Garrido, Josep Soler and Dold, Dominik and Frank, Johannes},
  booktitle={2021 IEEE International Conference on Cyber Security and Resilience (CSR)},
  pages={55--60},
  year={2021},
  organization={IEEE}
}
```

and 

```
@inproceedings{doldy2021energy,
  title={An energy-based model for neuro-symbolic reasoning on knowledge graphs},
  author={Doldy, Dominik and Garrido, Josep Soler},
  booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={916--921},
  year={2021},
  organization={IEEE}
}
```
