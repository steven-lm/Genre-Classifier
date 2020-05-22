

## Overview
This project was made as part of the Microsoft Student Accelerator program 2020.

I enjoy reading books in my spare time so i decided to create this project to automate the process of reading book blurbs to classify its genres. Multi label classification is also a prominent area of research in Machine Learning so I was intrigued with the challenge of combining text analysis and multi label prediction. 

I've also added explanations for the code in the Jupyter notebook.

**Approach:** 
To build a model that accurately predicts multiple book genres, we will be training three different models and comparing them. 
 - Random Forest Classifier
 - Logistic Regression Classifier (Incorporated with OneVsAll)
 - Feedfoward Neural Network

## Table of contents
* [Preprocessing](#preproccessing)
* [Training](#training)
   * [Baseline Model](#baseline-model)
  * [Random Forest](#random-forest)
  * [Logistic Regression](#logistic-regression)
  * [Neural Network](#neural-network)
* [Testing functions](#testing-functions)
* [Evaluation](#evaluation)
* [Setup & Dependencies](#setup)
*  [Practicality & Improvements](#future)

## Dataset:
**Source:** Carnegie Mellon University<br/>
**Name:** CMU Book Summary Dataset<br/>
**Books:** 16,559<br/>
**Metadata:**

 - Wiki ID
 - Freebase ID
 - Book Title
 - Book Author
 - Publication Date
 - Genres

## Preproccessing
### Data preprocessing:
**Extract useful data:** We will only be using 'Title', 'Genres' and 'Summary' for our models.<br/>
**Missing values:** Removed all missing values from our dataframe (3718 were found in the 'Genres' column).<br/>
**Freebase Tags:** To remove freebase tags, we import the data using json.loads<br/>
**Low frequency Genres:** Removed over 150 genres which contained less than 50 books, merged several subgenres into their respective genres (i.e Speculative Fiction -> Fiction).

![Genre Distribution](https://github.com/steven-lm/Genre-Classifier/blob/master/images/genredist.png)

### Text preprocessing:
**To clean the text features, we will:**

 1. Change all characters to lower case
 2. Remove any numbers from text
 3. Remove white spaces
 4. Remove punctuation (with String library)
 5. Remove words with less than 3 characters
 6. Remove stopwords (with NLTK)

Word distribution before:</br>
![Word Dist before](https://github.com/steven-lm/Genre-Classifier/blob/master/images/worddistbefore.png)

Word distribution after:</br>
![Word Dist after](https://github.com/steven-lm/Genre-Classifier/blob/master/images/worddistafter.png)


### Preparing data for models:
**Genres:** Since there are multiple genres to be classified, we will be using a multi label binarizer 

**Text data:** For the Logistic Regression and Random forest classifier, we will be using a TF-IDF vectorizer with a threshold of 0.8 and using a maximum of 10,000 features. 

For the Neural Network, we will create a work index using Keras's Tokenizer where the most frequent words will appear first. We will also convert the summaries into sequences and pad them to a max length of 500 characters. 

**Splitting Data:** We will be using the typical 80-20 train/test split to train our models.

## Training
To compare our models, we will be using their f1 scores which is the balance between precision/recall since there are multiple labels to be classified. We will also be using the **micro average** since there is a significant class imbalance. 
### Baseline model:
For our baseline model, we will simply take the most frequent genre (Fiction:   
4191) and since there are now 11282 books in our dataset, the baseline accuracy will be 4191/11282 = 37%. Note that accuracy is only with one genre so precision, recall and f1 score are not applicable. With that in mind this is purely for observation.
### Random Forest:
We will be comparing two Random Forest models, one by itself and one incorporated with OneVsRest.
![rd none](https://github.com/steven-lm/Genre-Classifier/blob/master/images/rdnone.png)

![enter image description here](https://github.com/steven-lm/Genre-Classifier/blob/master/images/rdone.png)

From the results, using the OneVsRest variation yields slightly better results.
### Logistic Regression:
Since Logistic Regression itself is binary, we must incorporate it with OneVsRest.</br>
![lr](https://github.com/steven-lm/Genre-Classifier/blob/master/images/f1_lr.JPG)

### Neural Network:
**Hyperparameters:** To find the optimal hyper parameters for our Neural Network, we will be utilising a combination of trial and error and Gridsearch.</br>
![grid](https://github.com/steven-lm/Genre-Classifier/blob/master/images/grid.png)

**Final Model Visual:**
![Model](https://github.com/steven-lm/Genre-Classifier/blob/master/images/NN.jpg)

**f1 score:**</br>
![f1_nn](https://github.com/steven-lm/Genre-Classifier/blob/master/images/f1_nn.png)

## Testing Functions
We will create two testing functions to observe our model.

**Inference function:** Our first function will take in a title name and search if it exists in our dataset. If it exists, this function will then use the summary of the book and predict its genres. We can then observe the difference between the prediction/actual genres. 
![Inference](https://github.com/steven-lm/Genre-Classifier/blob/master/images/inference_demo.png)

**Analyse function:** The analyse function takes in our own text and will predict its genres. For this example we will be using the summary of 'The Woman in White' (from Wikipedia) which does not exist in our dataset. 
![Analyse](https://github.com/steven-lm/Genre-Classifier/blob/master/images/analyse_demo.png)

The actual genres for this book are Novel, Fiction, Gothic and Mystery so this prediction is considerably accurate. 
	
## Evaluation
From the results, the Neural network achieved the best f1 score and if we were to measure its accuracy: </br>
![traintest_nn](https://github.com/steven-lm/Genre-Classifier/blob/master/images/NNTT.png)
![nn_acc](https://github.com/steven-lm/Genre-Classifier/blob/master/images/nn_accuracy.png)

we can assume that it is fairly effective.

In comparison to other similar multi classifiers such as [this movie genre predictor](https://github.com/igblackadder/movie_prediction/blob/master/genre%20prediction%20model.ipynb) which also achieved an f1 score in the 50s range, we can conclude that our Neural network is relatively accurate. 

## Setup 
The libraries/imports are located at the beginning of the Jupyter notebook. For the versions used at the time of creation of this project, refer to the requirements.txt file. 

## Future
**Practicality :** This model can be scaled up to create a book recommendation system based on preferred genres/previously read genres. Other models such as reading history and book ratings could be integrated to achieve this.

**What else could be tried?:** 

 - A better dataset could be used with a better genre distribution since this dataset contained almost 40% fiction.
 - A bigger dataset with more genres may produce a model which is able to predict rare rgenres
 - More models could be created and tested such as Recurrent Neural Networks or Long short-term memory Neural Networks
 - Words could further be cleaned using n-grams and stemming/lemmatization if we had more processing power
 - More word embedding techniques such as GloVe
