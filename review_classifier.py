import html, re, string, gzip, time, json
from pprint import pprint
import numpy as np
from scipy.stats import randint
import argparse
import multiprocessing

import tensorflow as tf

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from random import shuffle

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import cross_validation, grid_search
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

# random seed
seed = 7
cores = multiprocessing.cpu_count()

regex = re.compile('[%s]' % re.escape(string.punctuation))

class LabeledLineSentence(object):
    # this class was taken from the turorial in https://linanqiu.github.io/2015/10/07/word2vec-sentiment/
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

# write a quick function that parses the datafile
def parse(path):
  with gzip.open(path, 'r') as g:
    for l in g:
        yield eval(l)

# this was only a test function used to debug
def test_parse():
    for i,review in enumerate(parse('reviews_Electronics_5.json.gz')):
        print(int(review['overall']))
        print(type(review['overall']))
        time.sleep(5)

def create_json():
    #here, I created a full json file, where the reviews have been edited (no punctuation, lower case)
    f = open("reviews_strict.json", 'w')
    for l in parse('reviews_Electronics_5.json.gz'):
        l['reviewText'] = regex.sub('', html.unescape(l['reviewText'])).lower()
        l = json.dumps(l)
        f.write(l + '\n')
    f.close()


def review_lines(train_size,test_size):
    # here I create a document where each line is a review in order to train Doc2Vec
    # in order to get this done in finite time, I had to only go through the first O(10^5)-O(10^6) reviews.
    # I pick random documents from the corpus
    f = open("train.txt", 'w')
    g = open("test.txt", 'w')
    h = open("train_target.txt",'w')
    q = open("test_target.txt",'w')
    limit = 300000
    shuffle_indices = np.random.permutation(np.arange(limit))
    for i,review in enumerate(parse('reviews_Electronics_5.json.gz')):
        if i in shuffle_indices[:train_size]:
            if int(review['overall']) == 5 or int(review['overall']) == 4:
                review = regex.sub('', html.unescape(review['reviewText'])).lower()
                f.write(review + '\n')
                h.write('1' + '\n')
            elif int(review['overall']) == 1 or int(review['overall']) == 2:
                review = regex.sub('', html.unescape(review['reviewText'])).lower()
                f.write(review + '\n')
                h.write('0' + '\n')

        elif i in shuffle_indices[train_size:train_size + test_size]:
            if int(review['overall']) == 5 or int(review['overall']) == 4:
                review = regex.sub('', html.unescape(review['reviewText'])).lower()
                g.write(review + '\n')
                q.write('1' + '\n')
            elif int(review['overall']) == 1 or int(review['overall']) == 2:
                review = regex.sub('', html.unescape(review['reviewText'])).lower()
                g.write(review + '\n')
                q.write('0' + '\n')
        elif i > limit:
            break

    f.close()
    g.close()
    h.close()
    q.close()
    print('Completed generating test and training data...')

def d2v_source(train_size):
    # here I create a document where each line is a review in order to train Doc2Vec
    # in order to get this done in finite time, I had to only go through the first O(10^5)-O(10^6) reviews. This can be increased to increase performance
    f = open("d2v_train.txt", 'w')
    limit = 300000
    shuffle_indices = np.random.permutation(np.arange(limit))
    for i,review in enumerate(parse('reviews_Electronics_5.json.gz')):
        if i in shuffle_indices[:train_limit]:
            if int(review['overall']) == 5 or int(review['overall']) == 4:
                review = regex.sub('', html.unescape(review['reviewText'])).lower()
                f.write(review + '\n')
            elif int(review['overall']) == 1 or int(review['overall']) == 2:
                review = regex.sub('', html.unescape(review['reviewText'])).lower()
                f.write(review + '\n')
        elif i > limit:
            break

    f.close()

def create_doc2vec_model(vectorsize):
    # this creates the Doc2Vec model from 
    sources = {'d2v_train.txt':'TRAIN'} #,'test.txt':'TEST' }
    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(min_count=1, window=10, size=vectorsize, sample=1e-4, negative=5, workers=cores,alpha=0.025, min_alpha=0.025)
    model.build_vocab(sentences.to_array())

    print('Starting to train...')
    for epoch in range(10):
        print('Epoch ',epoch)
        model.train(sentences.sentences_perm()) # this is done so that SGD (stochastic gradient descent) can meanigfully converge

    model.save('./amzn.d2v')

    return model


# print(model.most_similar('good'))

def transform_input(vectorsize):
    # this loads the premade model saved as amzn.d2v and transforms writes its vectors into arrays that can be input into the scikit learn algorithms
    print('Loading Doc2Vec model...')
    try:
        model = Doc2Vec.load('./amzn.d2v')
    except Exception as exception:
        print('No existing model found. Starting to create a model...')
        train_size = 50000
        d2v_source(train_size)
        model = create_doc2vec_model(vectorsize)

    # load or generate train and test data
    try:
        with open('train.txt') as f:
            train_raw = np.asarray([line.rstrip('\n') for line in f])
        with open('test.txt') as f:
            test_raw = np.asarray([line.rstrip('\n') for line in f])
        with open('train_target.txt') as f:
            target = np.asarray([int(line.rstrip('\n')) for line in f])
        with open('test_target.txt') as f:
            target_test = np.asarray([int(line.rstrip('\n')) for line in f])
    
    except Exception as exception:
        print('No train data found. Generating new train and test files....')
        train_size = 50000
        test_size = 20000
        review_lines(train_size,test_size)
        with open('train.txt') as f:
            train_raw = np.asarray([line.rstrip('\n') for line in f])
        with open('test.txt') as f:
            test_raw = np.asarray([line.rstrip('\n') for line in f])
        with open('train_target.txt') as f:
            target = np.asarray([int(line.rstrip('\n')) for line in f])

        with open('test_target.txt') as f:
            target_test = np.asarray([int(line.rstrip('\n')) for line in f])

    # infer vectors for the sentences of the train and test sets
    # I do this by creating a list of strings out of the document and then converting that into a vector
    # this takes forever...so for further use, I will only do this for new train and test sets and save the vectors
    try:
         train_arrays = np.loadtxt('train_vectors.txt')
         test_arrays = np.loadtxt('test_vectors.txt')
    except Exception as exception:
    
        train_arrays = np.zeros((target.shape[0],vectorsize))
        test_arrays = np.zeros((target_test.shape[0],vectorsize))

        print('Vectorizing the train and test data...')

        for i in range(target.shape[0]):
            train_arrays[i,:] = model.infer_vector(train_raw[i].split())

        for i in range(target_test.shape[0]):
            test_arrays[i,:] = model.infer_vector(test_raw[i].split())

        np.savetxt('train_vectors.txt',train_arrays)
        np.savetxt('test_vectors.txt',test_arrays)

    return train_arrays, target, test_arrays, target_test


def train_ML_model(M_var,vectorsize):
    # in this function we define the logistic regression as the default choice for our machine learning algorithm. Other optional options are 
    # random forest classifier or gradient boosting classifier with the flag -M
    train, target, test, target_test = transform_input(vectorsize)

    if M_var == 'lr':
        classifier = LogisticRegression()
        print('Logistic regression chosen...')
    if M_var == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=2,oob_score=True, min_samples_split=3,min_samples_leaf=4 )
        print('Random Forest chosen...')
    if M_var == 'gb':
        classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.6, subsample=0.9, max_depth=1, random_state=0)
        print('Gradient Boosting chosen...')
    if M_var == 'nn':
        print('Neural network chosen...')
        neural_network(train, target, test, target_test,vectorsize)

    if M_var in ['lr','rf','gb']:
        classifier.fit(train, target)
        print('The model has a %s test score' % classifier.score(test,target_test))


def batch_iter(input_data, target, batch_size, shuffle=True):

    # generates an iterator over batches of th training data
    data_size = np.shape(input_data)[0]
    num_batches_per_epoch = int(data_size/batch_size) + 1

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = input_data[shuffle_indices]
        shuffled_target = target[shuffle_indices]
    else:
        shuffled_data = input_data
        shuffled_target = target

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index], shuffled_target[start_index:end_index]

def create_neural_network_model(data,keep_hidden,vectorsize):

    # the input dimension depends on the size of the doc2vec vectors
    n_input = vectorsize

    n_hidden = int(1.5*vectorsize)
    n_hidden2 = int(1.5*vectorsize)

    hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([n_input,n_hidden])), 'biases' : tf.Variable(tf.random_normal([n_hidden]))}
    # hidden_layer2 = {'weights' : tf.Variable(tf.random_normal([n_hidden,n_hidden2])), 'biases' : tf.Variable(tf.random_normal([n_hidden2]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_hidden,2])), 'biases' : tf.Variable(tf.random_normal([2]))}

    layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, keep_hidden)

    # layer2 = tf.add(tf.matmul(layer1,hidden_layer2['weights']),hidden_layer2['biases'])
    # layer2 = tf.nn.relu(layer2)
    # layer2 = tf.nn.dropout(layer2, keep_hidden)

    output = tf.add(tf.matmul(layer1,output_layer['weights']),output_layer['biases'])

    return output


def neural_network(train, target, test, target_test,vectorsize):

    # this initializes and trains the neural network

    x = tf.placeholder('float',[None,vectorsize]) #data
    y = tf.placeholder(dtype=tf.int64)
    keep_hidden = tf.placeholder('float',)
    
    prediction = create_neural_network_model(x,keep_hidden,vectorsize)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction,y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 20
    batch_size = 100

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            batches = batch_iter(train, target, batch_size)
            epoch_loss = 0
            for batch in batches:
                x_batch, y_batch = batch
                _, c = sess.run([optimizer, cost], feed_dict={x : x_batch, y : y_batch, keep_hidden : 0.7})
                epoch_loss += c
            print('Epoch', (epoch+1), 'completed out of', n_epochs, 'loss:', "%.3f" % epoch_loss)


        correct = tf.equal(tf.argmax(prediction, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x : test, y : target_test, keep_hidden : 1.0}))


def main():

    # size of the word vectors
    vectorsize = 200
    # number of documents that are used to create the Doc2Vec model
    train_size = 50000

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-M", dest = "M_input", help="ML method", type=str, choices=['lr', 'rf', 'gb', 'nn'])
    parser.add_argument("-r",default = False, dest="remake",  help='create new Doc2Vec model, default: no', type=bool, choices=[True])
    args = parser.parse_args()

    if args.M_input:
        M_var = args.M_input
    else:
        M_var = 'lr'

    if args.remake:
        remake_doc2vec = bool(args.remake)
    else:
        remake_doc2vec = False

    if remake_doc2vec:
        print('Generating a new Doc2Vec model...')
        d2v_source(train_size)
        create_doc2vec_model(vectorsize)

    train_ML_model(M_var,vectorsize)

if __name__ == '__main__':

    main()









