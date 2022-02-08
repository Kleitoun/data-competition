import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay

train = pd.read_csv('train.csv').drop(columns=['S.No'])
test = pd.read_csv('test.csv').drop(columns=['S.No'])

class LogisticRegression:

    def __init__(self,train,test):
        self.train = train.to_numpy()
        self.test = test.to_numpy()
        self.val = None
        self.val_y = None
        self.features = None
        self.targets = None
        self.weights = np.ones((len(self.train[0]),3))

    def normalize(self):
        mu = self.train[:,:-1].mean(axis=0)
        sigma = self.train[:,:-1].std(axis=0)
        self.train[:,:-1] = (self.train[:,:-1]-mu)/sigma
        self.test = (self.test-mu)/sigma

    def preprocess(self):
        self.test = np.c_[self.test,np.ones(len(self.test))]
        index = np.arange(len(self.train))
        np.random.shuffle(index)
        train_index = index[:int(0.8*len(self.train))]
        val_index = index[int(0.8*len(self.train)):]
        self.features = self.train[train_index,:-1]
        self.targets = self.train[train_index,-1]
        self.val = self.train[val_index,:-1]
        self.val_y = self.train[val_index,-1]
        self.features =  np.c_[self.features,np.ones(len(self.features))]
        self.val = np.c_[self.val,np.ones(len(self.val))]

    def one_hot_encoding(self):
        temp = [0,1,2]
        pred = np.zeros((len(self.targets),3))
        for i in range(3):
            for j in range(len(self.targets)):
                if self.targets[j] == temp[i]:
                    pred[j,i] = 1
        self.targets = pred.reshape(-1,3)

    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)

    def gradient(self,X,y):
        z = X @ self.weights
        pred = self.softmax(z.reshape(-1,3))
        gd = ((1/len(self.features)) * (X.T @ (pred-y)))
        return gd

    def fit(self,max_iters=10000,lr=0.01):
        for i in range(max_iters):
            self.weights -= lr * self.gradient(self.features,self.targets)

    def validate(self):
        z = self.val @ self.weights
        pred = np.argmax(self.softmax(z.reshape(-1,3)),axis=1)
        count = 0
        for i in range(len(self.val)):
            if pred[i]!=np.argmax(self.val_y[i]):
                count+=1
        print(classification_report(pred,self.val_y))
        disp = ConfusionMatrixDisplay(confusion_matrix(self.val_y,pred),display_labels=np.unique(self.val_y))
        disp.plot()
        plt.show()
        return 1-(count/len(self.val))

    def visualize(self):
        z = self.features @ self.weights
        pred = np.argmax(self.softmax(z.reshape(-1,3)),axis=1)
        count = 0
        for i in range(len(self.features)):
            if pred[i]!=np.argmax(self.targets[i]):
                count+=1
        return 1-(count/len(self.features))

    def predict(self):
        z = self.test @ self.weights
        pred = np.argmax(self.softmax(z.reshape(-1,3)),axis=1)
        return pred

P = LogisticRegression(train,test)
P.normalize()
P.preprocess()
P.one_hot_encoding()
P.fit()
print(P.validate())
print(P.visualize())
predictions = P.predict()
df = pd.DataFrame({'LABELS':predictions})
df.reset_index(inplace=True)
df.rename(columns={'index':'S.No'},inplace=True)
df.to_csv('predictions_log_reg_mc.csv',index=False)
print(df.LABELS.value_counts())
