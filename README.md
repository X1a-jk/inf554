# Introduction
Social networking is developing rapidly these days and many people usually express their opinions on sites such as Twitter and so on. Some posts get retweeted by many people, while some tweets get very few retweets. The number of retweets has always been a topic of great interest to the public. In this project, based on a new Twitter dataset centred on the topic of the 2022 French presidential election, we build a model to predict the number of retweets a tweet will receive based on some of its characteristics.

# Data
We used the dataset provided by the kaggle competition, the link of the sataset is given as below: https://www.kaggle.com/competitions/retweet-prediction-challenge-2022/data. 
Training and validation data: "data/train.csv"
Evaluation data: "data/evaluation.csv"

# Embedding
We used a French vocabulary dictionary to get the embedding of the tweets written in French, the link of witch is given as below: https://github.com/Ismailhachimi/French-Word-Embeddings/tree/master/Data. 
Dictionary file: "French-Word-Embeddings/Data/data.txt"

To train the embedding model, you have to uncomment the lines below 
```python
# model_text10=train_text(X_documents, size=64)
# model_text10.save('WE_models/d2v_64D')
# model_text5=train_text(X_documents, size=32)
# model_text5.save('WE_models/d2v_32D')
```

or you can load the pretrained embedding files saved under "WE_models" by running these lines
```python
model_text10 = Doc2Vec.load('WE_models/d2v_10D')
model_text5 = Doc2Vec.load('WE_models/d2v_5D')
```

# Environment:
In the project, sevral modules are used. Their requirements are listed as follow:
* numpy==1.23.1
* pandas==1.5.1
* scikit-learn==1.0.1
* verstack==3.2.4
* nltk==3.7
* torch==1.13.0
* torchvision==0.14.0
* matplotlib==3.5.2
* matplotlib-inline==0.1.6
* scipy==1.9.3
* gensim==4.2.0

The environment is also saved in the environment.yaml, which can be directly imported by using anaconda.

# How to run the file 
There are three jupyter notebooks, different feature embeddings and network structures are applied in each of them

projet.ipynb: For the hashtag features, the length of their corresponding list is used as the feature and a net with two FC layers is trained, this is the early version of the project witch is then improved. 
project_vec_lstm.ipynb: For the hashtag features, we have applied the embedding algorithm to the contents, and a net with LSTM layers is trained.
project_vec.ipynb: For the hashtag features, we have applied the embedding algorithm to the contents, and a net with convolution layers is trained.

You can simply run the notebooks from top to bottom. For the embedding models, you cai either train a model on your own or load the existing models as said above.

You can also find the projet on github: https://github.com/xiajunkai328/inf554 where the files all already placed in the right places. 