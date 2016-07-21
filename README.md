AMZNREVS
=============


This is a small program that used Doc2Vec to vectorize user reviews for products (for example on Amazon) and subsequently use sentiment analysis in order to classify the review as either good or bad. 

### How to use

So far, the program relies on provided review data in order to train the model for classification. A pretrained model is included in the repository, but it can easily be replaced by a new model using the flag -l True. The train dataset consists of 60000 either positive or negative reviews and the test data consists of additional 20000 reviews.

The -m flag can be used to select different machine learning algorithms. The options are

- lr: logistic regression (default)

- rf: Random forest

- gd: Gradient boosting

