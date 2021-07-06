# Data Preprocessing

## Dataset

The EUVP dataset can be downloaded from [here](http://irvlab.cs.umn.edu/resources/euvp-dataset). It contains several sets of paired data and unpaired data of good and poor quality. 

![data](../images/data.png)

The folder structure is shown as follows:

```
Paired/
├── underwater_imagenet/
│   ├── trainA/
│   ├── trainB/
│   └── validation/
├── underwater_dark/
│   ├── trainA/
│   ├── trainB/
│   └── validation/
└── underwater_scenes/
    ├── trainA/
    ├── trainB/
    └── validation/
Unpaired/
├── trainA/
├── trainB/
└── validation/
test_samples/
├── GTr/
└── Inp/
```

### Paired Data

Three sets(`underwater_imagenet`, `underwater_dark`, and `underwater_scenes`) of paired data are included. The statistics show as below:

|                     | Training Pairs | Validation |
|---------------------|----------------|------------|
| Underwater Imagenet | 3700           | 1270       |
| Underwater Dark     | 5550           | 570        |
| Underwater Scenes   | 2185           | 130	    |

However, the validation data are not in pairs. In our work, 90% of total pairs are used as training data and 10% of total pairs are used as validation data. Data splitting is executed using `data_split.py`. `-p` is specified to create paired dataset. 

```shell
python data_split.py -d "/path/to/EUVP/Paired/underwater_imagenet" -p
python data_split.py -d "/path/to/EUVP/Paired/underwater_dark" -p
python data_split.py -d "/path/to/EUVP/Paired/underwater_scenes" -p
```

`splits.json` is created in the specified folder `-d` and has the format as below:

```json
{
    "train": ["img1.jpg", "img4.jpg", ..., "img101.jpg"],
    "valid": ["img2.jpg", "img3.jpg", ..., "img100.jpg"]
}
```
After paired data splitting, the train/validation statistics are as follows:

|                     | Train | Validation | Total Pairs |
|---------------------|-------|------------|-------------|
| Underwater Imagenet | 3330  | 370        |3700         |
| Underwater Dark     | 4995  | 555        |5550         |
| Underwater Scenes   | 1967  | 218        |2185         |

In the folder `test_samples`, total 515 data are in pairs. Images in `GTr` are in good quality and images in `Inp` are in poor quality.

### Unpaired Data

The original unpaired data statistics show as below:

| Good Quality | Poor Quality | Validation |
|--------------|--------------|------------|
| 3140         | 3195         | 330        |

90% of good(in folder `trainB`) and poor(in folder `trainA`) quality images are used as training data and left 10% of images are used as validation data. Data splitting is executed using `data_split.py` too.

```shell
python data_split.py -d "/path/to/EUVP/Unpaired"
```

`splits.json` is created in the specified folder `-d` and has the format as below:

```json
{
    "train": ["trainA/img1.jpg", "trainB/img4.jpg", ..., "trainA/img101.jpg"],
    "valid": ["trainA/img2.jpg", "trainB/img3.jpg", ..., "trainB/img100.jpg"]
}
```
After paired data splitting, the train/validation statistics are as follows:

|        | Train | Validation | Total Images |
|--------|-------|------------|--------------|
| trainA | 2885  | 320        | 3205         |
| trainB | 2826  | 314        | 3140         |