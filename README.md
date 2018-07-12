# CWS_LSTM
Code for EMNLP2015 paper **Long Short-Term Memory Neural Networks for Chinese Word Segmentation**.

Note
========
This project fork from [CWS_LSTM](https://github.com/FudanNLP/CWS_LSTM), its original author is Doctor [Xinchi Chen](https://github.com/dalstonChen).

I add some data process logic to run the code.

Get start
=============

download the training data from [link](https://github.com/supercoderhawk/CWS_LSTM/releases/download/v0.0.1-data/msr.zip), and unzip them into `data/` directory (require to create manually).

execute the following command to run the code
```bash
pip install -r requirements.txt   # install requirements
cd src/
python data_process.py   # initialize directory and data
python driver.py    # start to training
```
