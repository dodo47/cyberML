# Machine learning on knowledge graphs for context-aware security monitoring

This repository contains the dataset used in the publication "Machine learning on knowledge graphs for context-aware security monitoring".

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
