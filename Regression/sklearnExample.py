#!/usr/bin
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


'''Linear Models'''

"""Linear Regression"""
# 获取数据集
diabets_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabets_X = diabets_X[:, np.newaxis, 2] # np.newaxis 对数组增加一个维度

diabets_X_train = diabets_X[: -20]
diabets_X_test = diabets_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 建模
regr = linear_model.LinearRegression()
regr.fit(diabets_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabets_X_test)

print('Coef_:', regr.coef_)
print('Mean squared error:%.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))

'''
plt.scatter(diabets_X_test, diabetes_y_test, color='black')
plt.plot(diabets_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

# plt.show()
'''

"""Ridge regression and classification"""
# 增加了惩罚性， 系数使残差最小化 岭回归鲁棒性更强

X = 1. / (np.arange(1, 11)) + np.arange(0, 10)[:, np.newaxis]
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
'''
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
print(alphas)
'''

""" classification of text documents using sparse features"""

import logging
from optparse import OptionParser
import sys
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
op = OptionParser()
op.add_option("--report", action="store_true", dest="print_report",
              help="print a detailed classification report.")
op.add_option("--chi2_select", action="store", type="int", dest="select_chi2",
              help="select some number of features using a chi-squared test")
op.add_option("--confusion_matrix", action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)
print(__doc__)
op.print_help()
print()

# 从指定主题目录获取数据集
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes') # 需要过滤的可能导致模型过拟合的子集
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")


data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

target_names = data_train.target.names #返回结果的分类标签

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)
print("%d documents - %0.3fMB (traing set)" %(len(data_train.data), data_test_size_mb))
print("%d documents - %0.3fMB (test set)" %(len(data_test.data), data_test_size_mb))
print("%d categories" % len(target_names))

y_train, y_test = data_train.target, data_test.target

print(("Exracting features from the traing data using a sparse vectorizer"))
t0=time()
# 提取训练集特征
if opts.use_hasing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)

duration = time()-t0

print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

# 提取测试集特征
print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()


if opts.use_hasing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

