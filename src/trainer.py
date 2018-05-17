from dataloader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
from numpy.random import *
from sklearn.metrics import accuracy_score


def generate_data(X_train,y_train):
        for i in range(10):
            ind1= randint(0,359,3)
            ind2 =randint(360,399,3)
            
            x1 = X_train[ind1]
            x2 = X_train[ind2]
            y1 = y_train[ind1]
            y2 = y_train[ind2]
            
            x = np.concatenate([x1,x2],axis=0)
            y = np.concatenate([y1,y2])
            
            yield (x,y)

def training(model, X_train,y_train,X_test,y_test):
        for ite in range(91):
            for x, y in generate_data(X_train,y_train):
                model.train_on_batch(x, y)
            if ite%10==0:
                print('Iter{} validation acc'.format(ite),accuracy_score(y_test,get_class(model.predict(X_test))))
                pass
        return model
    
def get_class(pred):
        y_pred=[]
        for f in pred:
            if f>0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)

def plot_confusion_matrix(y_test, pred_y,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        plt.figure()
        classes=['NG','OK']
        cm = confusion_matrix(y_test, pred_y)
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()

  
