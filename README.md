AMZNREVS
=============


This is a small program that used Doc2Vec to vectorize user reviews for products (for example on Amazon) and subsequently use sentiment analysis in order to classify the review as either good or bad. 

### How to use

So far, the program relies on provided review data in order to train the model for classification. The train dataset consists of 60000 either positive or negative reviews and the test data consists of additional 20000 reviews. When using for the first time, the program should detect no existing model and should automatically create the Doc2Vec model based on the provided train and test data. Afterwards this can be forced by using the -l True flag in order to make a new Doc2Vec model. 

In addition, the vectorization of the train and test files will only be done in the first run. Afterwards it will be saved in two text files in order to save computation time (at the cost of space on the hard drive).

The -m flag can be used to select different machine learning algorithms. The options are

- lr: logistic regression (default)

- rf: Random forest

- gd: Gradient boosting

- nn: Neural network with one hidden layer with dropout

Most of the classification algorithms have an accurary of about 89-90%.

### Future additions

- implement a scraping mechanism that will provide new reviews in order to classify with the trained model

- add summary method to summarize new reviews and get important points

