import csv
import pandas as p
import string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


messages = p.read_csv('/Users/devashreeshirude/Desktop/reviews.csv', sep='|', names=['labels', 'text'])
messages = messages.drop([0])
length = len(messages)
df1 = list(range(4, length, 5))
messages1 = messages.ix[df1]
messages = messages.drop(df1)
# '''
# messages1 = p.read_csv('/Users/laxmipoojaanegundi/Desktop/HW2/test3.csv',sep='|',names=['text'])
# df=messages[messages.columns[1]]
#
# df = df.str.lower()
#
# def text_process(df):
#     nopunc = [char for char in df if char not in string.punctuation]
#     nopunc = ''.join(nopunc)
#     return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
# df = df.apply(text_process)
#
# '''

messages = messages.dropna(subset=['text'])
messages1 = messages1.dropna(subset=['text'])

count_vect = CountVectorizer()
count_vect1 = count_vect

X_train_counts = count_vect.fit_transform(messages['text'])
Y_train_counts = count_vect1.transform(messages1['text'])
print(X_train_counts.shape)
print(Y_train_counts.shape)

# # ----------------SVM----------------
# count_vect3 = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_df=.80, min_df=4)
# count_vect4 = count_vect3
# X_train_counts1 = count_vect3.fit_transform(messages['text'])
# Y_train_counts1 = count_vect4.transform(messages1['text'])
# classifier_rbf = LinearSVC(random_state=0)
# classifier_rbf.fit(X_train_counts1, messages['labels'])
# prediction_rbf = classifier_rbf.predict(Y_train_counts1)
# print("For the SVM classifier:")
# print("Accuracy for SVM")
# print(accuracy_score(messages1['labels'], prediction_rbf))
# cm = confusion_matrix(messages1['labels'], prediction_rbf)
# print('Confusion Matrix :',cm)

#---FinalResults---#
#print("Report for viewing:",classification_report(messages1['labels'], prediction_rbf))
#
# # # ---------------MLP Classifier----------------------------------
# clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(8,5), random_state=1,early_stopping = True)
# #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state=1,hidden_layer_sizes=(1000,))
# # clf = MLPClassifier(alpha=1e-5, random_state=1, hidden_layer_sizes=(1000,))
# #clf = MLPClassifier(alpha=1e-5, random_state=1, hidden_layer_sizes=(1000,), learning_rate='adaptive')
# clf.fit(X_train_counts, messages['labels'])
# print("For the MLP classifier:")
# prediction_rbf = clf.predict(Y_train_counts)
# print("For the accuracy of  MLP")
# print(accuracy_score(messages1['labels'], prediction_rbf))

# count_vect3 = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_df=.80, min_df=4)
# count_vect4 = count_vect3
# X_train_counts1 = count_vect3.fit_transform(messages['text'])
# Y_train_counts1 = count_vect4.transform(messages1['text'])
# NB = MultinomialNB()
# NB.fit(X_train_counts1, messages['labels'])
# prediction_rbf = NB.predict(Y_train_counts1)
# print("For the accuracy of  NB")
# print(accuracy_score(messages1['labels'], prediction_rbf))
# cm = confusion_matrix(messages1['labels'], prediction_rbf)
# print('Confusion Matrix :',cm)
#
# #---FinalResults---#
# print("Report for viewing:",classification_report(messages1['labels'], prediction_rbf))

count_vect3 = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_df=.80, min_df=4)
count_vect4 = count_vect3
X_train_counts1 = count_vect3.fit_transform(messages['text'])
Y_train_counts1 = count_vect4.transform(messages1['text'])
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train_counts1, messages['labels'])
prediction_rbf = clf.predict(Y_train_counts1)
print("For the Random classifier:")
print (prediction_rbf)
print("For the accuracy for RandomForest")
print(accuracy_score(messages1['labels'], prediction_rbf))
cm = confusion_matrix(messages1['labels'], prediction_rbf)
print('Confusion Matrix :',cm)

#---FinalResults---#
print("Report for viewing:",classification_report(messages1['labels'], prediction_rbf))