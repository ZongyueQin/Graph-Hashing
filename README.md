# Graph-Hashing

## How to get data?

Download the data from https://drive.google.com/open?id=1rF4QA-HDgNfFOme7M2Jw8JjYXNb6rCye and uncompress it under data directory.

## How to compile?

First enter *model/BSS-GED* and run **./setup.sh**
Then enter *main*, run **make**

## How to train a model?

First make sure configure config.py correctly. Then run **python TrainModel.py**

## How to run the benchmark?

Enter *main*.
Then run **bin/main** with correct parameters. The parameters to run **AIDS**, **linux15**, **ALCHEMY** in our experiment is shown below.

Dataset | command
--------|---------
AIDS | bin/main ../data/AIDS/train/graphs.bss 42587 ../data/AIDS/test/graphs.bss 100 15 model_path inverted_index_path (GED2Hamming mapping file)
linux15 | bin/main ../data/linux15/train/graphs.bss 24087 ../data/linux15/test/graphs.bss 100 15 model_path inverted_index_path (GED2Hamming mapping file)
ALCHEMY | bin/main ../data/ALCHEMY/train/graphs.bss 99676 ../data/ALCHEMY/test/graphs.bss 100 15 model_path inverted_index_path (GED2Hamming mapping file)

## Dependecies
* python3: Assume python3 by default (use pip3 to install packages).
* numpy
* pandas
* scipy
* scikit-learn
* tensorflow
* networkx==1.11 (NOT 2.1)
* beautifulsoup4
* lxml
* matplotlib
* seaborn
* colour
* pytz
* requests

Reference commands: sudo pip3 install numpy pandas scipy scikit-learn tensorflow networkx==1.10 beautifulsoup4 lxml matplotlib seaborn colour pytz requests

## Result Visualization
temporary link: [http://lemon.chiaolu.me/](http://lemon.chiaolu.me/)
