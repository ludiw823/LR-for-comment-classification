import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
import LR
import sklearn.metrics as sk
from sklearn.utils import shuffle
#df = shuffle(df) #random shuffle the datset

data1 = pd.read_csv('/Users/ludi/Desktop/Machine Learning/project/machine8000.csv',encoding="ISO-8859-1")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

datay=data1[label_cols]
y, testy = train_test_split(data1, test_size=0.2, random_state=42)
data=np.loadtxt('/Users/ludi/Desktop/Machine Learning/project/a8000.txt',delimiter = ',')
x, testx = train_test_split(data, test_size=0.2, random_state=42)    


'''
#For TF-IDF feature

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

#train['none'] = 1-train[label_cols].max(axis=1)
#train.describe()

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import re, string
re_tok = re.compile('([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, max_features = 2000)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
print('trn_term_doc',x.shape)

test_x = test_term_doc
print('test_term_doc',test_x.shape)
'''
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LR.LogisticRegression()
    #x_nb = x.multiply(r)
    x_nb = x*r
    m.initialize_weights(x_nb.shape[1])
    print('start')
    print('x_nb',x_nb.shape)
    print(type(x_nb))
    print('y',y.shape)
    print('end')
    print(x_nb.data)
    #x_nb_arr = x_nb.A
    x_nb_arr = x_nb
    #x_nb_arr = x_nb_arr.reshape(x_nb.shape)
    #x_nb to the type array
    print('x_nb_arr',x_nb_arr.shape)
    
    
    m.fit(x_nb_arr, y)
    return m, r

#use NBLR
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(y[j])
    name ='preds'+str(j)
    locals()['preds'+str(j)]=m.predict(testx*r)
    #preds = m.predict(test_x.multiply(r).A)

    
pred_all = np.hstack([predstoxic,predssevere_toxic,predsobscene,predsthreat,predsinsult,predsidentity_hate])
#print(pred_all)

#print accuracy for each class
acctoxic = sk.accuracy_score(testy['toxic'],predstoxic)
print('acctoxic', acctoxic)

accsevere_toxic = sk.accuracy_score(testy['severe_toxic'],predssevere_toxic)
print('accsevere_toxic', accsevere_toxic)

accobscene = sk.accuracy_score(testy['obscene'],predsobscene)
print('obscene', accobscene)

accthreat = sk.accuracy_score(testy['threat'],predsthreat)
print('threat', accthreat)

accinsult = sk.accuracy_score(testy['insult'],predsinsult)
print('insult', accinsult)

accidentity_hate = sk.accuracy_score(testy['identity_hate'],predsidentity_hate)
print('identity_hate', accidentity_hate)
