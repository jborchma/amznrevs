from amazon_scraper import AmazonScraper
import itertools
import html, re, string, gzip, time, json
from amazon.api import AmazonAPI
from pprint import pprint
import numpy as np
from scipy.stats import randint
import argparse

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from random import shuffle

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import cross_validation, grid_search
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

seed = 7

regex = re.compile('[%s]' % re.escape(string.punctuation))

access_key_id = ""
secret_access_key = ""

# for USA
assoc_tag_us = ""

amzn = AmazonScraper(access_key_id, secret_access_key, assoc_tag_us,region="US")

prod = amzn.search(Keywords='Samsung UN55JS8500',SearchIndex='Electronics')#Electronics

#print(first_five[0].reviews_url)

# the reviews still have the html codes for certain plain text symbols that need to be decoded if we want to use them, e.g. &=34 for ".
# for that purpose we use html.unescape on the text of the review.

# for p in itertools.islice(prod, 1):
#     print(p.asin)
#     rs = amzn.reviews(ItemId=p.asin) #look up the reviews for the itemid of the item in question. This way we get them all.
#     revs = rs.full_reviews()
#     for r in revs:
#         print(html.unescape(r.text)) #prints the text of the review
#         #print(int(5*r.rating), 'stars') # prints the star rating
#         time.sleep(0.5)

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


def review_lines(train_limit,test_limit):
    # here I create a document where each line is a review in order to train Doc2Vec
    # in order to get this done in finite time, I had to only go through the first 30000 reviews. This can be increased to increase performance
    f = open("train.txt", 'w')
    g = open("test.txt", 'w')
    h = open("train_target.txt",'w')
    q = open("test_target.txt",'w')
    for i,review in enumerate(parse('reviews_Electronics_5.json.gz')):
        if int(review['overall']) == 5 or int(review['overall']) == 4:
            review = regex.sub('', html.unescape(review['reviewText'])).lower()
            if i <train_limit:
                f.write(review + '\n')
                h.write('1' + '\n')
            elif i <test_limit:
                g.write(review + '\n')
                q.write('1' + '\n')
            else:
                break
        elif int(review['overall']) == 1 or int(review['overall']) == 2:
            review = regex.sub('', html.unescape(review['reviewText'])).lower()
            if i <train_limit:
                f.write(review + '\n')
                h.write('0' + '\n')
            elif i<test_limit:
                g.write(review + '\n')
                q.write('0' + '\n')
            else:
                break
    f.close()
    g.close()
    h.close()
    q.close()

def create_doc2vec_model():
    # this creates the Doc2Vec model from 
    sources = {'train.txt':'TRAIN','test.txt':'TEST' }
    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(sentences.to_array())

    print('Starting to train...')
    for epoch in range(10):
        print(epoch)
        model.train(sentences.sentences_perm())

    model.save('./amzn.d2v')

    return model


# print(model.most_similar('good'))

def transform_input():
    # this loads the premade model saved as amzn.d2v and transforms writes its vectors into arrays that can be input into the scikit learn algorithms
    print('Loading Doc2Vec model...')
    try:
        model = Doc2Vec.load('./amzn.d2v')
    except Exception as exception:
        print('No existing model found. Starting to create a model...')
        train_limit = 40000
        test_limit = 60000
        review_lines(train_limit,test_limit)
        create_doc2vec_model()


    with open('train_target.txt') as f:
        target = np.asarray([line.rstrip('\n') for line in f])

    with open('test_target.txt') as f:
        target_test = np.asarray([line.rstrip('\n') for line in f])

    train_arrays = np.zeros((target.shape[0],100))
    test_arrays = np.zeros((target_test.shape[0],100))

    for i in range(target.shape[0]):
        train_arrays[i] = model.docvecs[i]

    for i in range(target_test.shape[0]):
        test_arrays[i] = model.docvecs[i+target.shape[0]]

    return train_arrays, target, test_arrays, target_test


def train_ML_model(M_var):
    # in this function we define the logistic regression as the default choice for our machine learning algorithm. Other optional options are 
    # random forest classifier or gradient boosting classifier with the flag -M
    train, target, test, target_test = transform_input()

    if M_var == 'lr':
        classifier = LogisticRegression()
        print('Logistic regression chosen...')
    if M_var == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=2,oob_score=True, min_samples_split=3,min_samples_leaf=4 )
        print('Random Forest chosen...')
    if M_var == 'gb':
        classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.6, subsample=0.9, max_depth=1, random_state=0)
        print('Gradient Boosting chosen...')

    classifier.fit(train, target)
    print('The model has a %s test score' % classifier.score(test,target_test))

# print('Create files...')
# review_lines(40000,60000)
# print('Files created...')


# print('Create model....')
# model = create_doc2vec_model()
# print('Model created....')

def store():

    kfolds = cross_validation.KFold(n=train.shape[0], n_folds = 3, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2,oob_score=True, min_samples_split=3,min_samples_leaf=4 )
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.6, subsample=0.9, max_depth=1, random_state=0)

    # param_dist = {'learning_rate':[0.1,0.2,0.4,0.6],'subsample':[0.3,0.5,0.7,0.9]}

    # random_search = GridSearchCV(clf, param_grid = param_dist, cv=3)
    # random_search.fit(train, target)
    # top_params = report(random_search.grid_scores_,3)

    # print(top_params)

    clf.fit(train, target)
    #results = cross_validation.cross_val_score(rf,train,target,cv=kfolds)
    print(clf.score(test,target_test))
    #print(results.mean())


def main():

    train_limit = 40000
    test_limit = 60000

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-M", dest = "M_input", help="ML method", type=str, choices=['lr', 'rf', 'gb'])
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

    print(remake_doc2vec)

    if remake_doc2vec:
        review_lines(train_limit,test_limit)
        create_doc2vec_model()

    train_ML_model(M_var)

if __name__ == '__main__':

    main()















