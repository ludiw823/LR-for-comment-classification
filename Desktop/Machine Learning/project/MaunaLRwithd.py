import pandas as pd, numpy as np
import LR
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk
from sklearn.utils import shuffle
import numpy
#df = shuffle(df) 随机排序

#load 8000 samples
data1 = pd.read_csv('/Users/ludi/Desktop/Machine Learning/project/machine8000.csv',encoding="ISO-8859-1")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#load label in datay
datay=data1[label_cols]
#sepArate y into train and test
y, testy = train_test_split(data1, test_size=0.2, random_state=42)

#load prepocessed feature matrix X
data=np.loadtxt('/Users/ludi/Desktop/Machine Learning/project/a8000.txt',delimiter = ',')
#separate X into train and test
#datax=np.array(data)
x, testx = train_test_split(data, test_size=0.2, random_state=42)    

#train
for i in label_cols:
    print('fit', i)
    m = LR.LogisticRegression()
    m.initialize_weights(x.shape[1])
    m.fit(x,y[i])
    name ='preds'+str(i)
    #predict
    locals()['preds'+str(i)]=m.predict(testx) 

#print accuracy for each class
pred_all = np.hstack([predstoxic,predssevere_toxic,predsobscene,predsthreat,predsinsult,predsidentity_hate])
testy = testy[label_cols]

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


#np.savetxt('label8000.txt', datay, fmt="%d",delimiter=",")


