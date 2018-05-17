import os
import librosa
import pickle
import numpy as np
import glob
from keras.preprocessing.sequence import pad_sequences

class DataLoader:
    """docstring for ClassName"""
    def __init__(self):
        self.data_root='../input_data'
        self.ok_train_path = os.path.join(self.data_root,'train/OK')
        self.ng_train_path = os.path.join(self.data_root,'train/NG')
        self.ok_test_path = os.path.join(self.data_root,'test/OK')
        self.ng_test_path = os.path.join(self.data_root,'test/NG')
        self.dev_path = os.path.join(self.data_root,'dev')
        self.pickle_path = os.path.join(self.data_root,'pickle')
    def data_num(self):
        def get_num(path):
            path = os.path.join(path,'*.wav')
            return len(glob.glob(path))
        print('trainOK',get_num(self.ok_train_path),'件')
        print('trainNG',get_num(self.ng_train_path),'件')
        print('testOK', get_num(self.ok_test_path),'件')
        print('testNG', get_num(self.ng_test_path),'件')
        print('dev',    get_num(self.dev_path),'件')
    def preprocess(self):
        def get_data(file_path):
            x, fs = librosa.load(file_path, sr=44100)
            return librosa.feature.mfcc(x, sr=fs)
        X_train_ok =[get_data(path) for path in glob.glob(self.ok_train_path+'/*.wav') ]
        y_train_ok =np.array([1 for _ in glob.glob(self.ok_train_path+'/*.wav')])
        print('OK train Done')

        X_test_ok =[get_data(path) for path in glob.glob(self.ok_test_path+'/*.wav')  ]
        y_test_ok =np.array([1 for _ in glob.glob(self.ok_test_path+'/*.wav')])
        print('OK test Done')

        X_train_ng =[get_data(path) for path in glob.glob(self.ng_train_path+'/*.wav')  ]
        y_train_ng =np.array([0 for _ in glob.glob(self.ng_train_path+'/*.wav')])
        print('NG train Done')

        X_test_ng =[get_data(path) for path in glob.glob(self.ng_test_path+'/*.wav')  ]
        y_test_ng =np.array([0 for _ in glob.glob(self.ng_test_path+'/*.wav')])
        print('NG test Done')

        dev_X =[get_data(path) for path in glob.glob(self.dev_path+'/*.wav')  ]
        print('dev Done')

        self.X_train = X_train_ok+X_train_ng
        self.y_train = np.concatenate([y_train_ok,y_train_ng])
        self.X_test  = X_test_ok+X_test_ng
        self.y_test  = np.concatenate([y_test_ok,y_test_ng])
        self.X_dev   = dev_X

        l0 = np.max([_.shape[1] for _ in self.X_train])
        l1 = np.max([_.shape[1] for _ in self.X_test])
        l2 = np.max([_.shape[1] for _ in self.X_dev])
        
        maxlen = np.max([l0,l1,l2])
        
        self.X_train = [x.transpose() for x in self.X_train]
        self.X_test  = [x.transpose() for x in self.X_test]
        self.X_dev  = [x.transpose() for x in self.X_dev]


        self.X_train = pad_sequences(self.X_train, maxlen=maxlen)
        self.X_test  = pad_sequences(self.X_test, maxlen=maxlen)
        self.X_dev   = pad_sequences(self.X_dev, maxlen=maxlen)


    def save_pickle(self):

        with open(os.path.join(self.pickle_path,'X_train.pickle'), 'wb') as f:
            pickle.dump(self.X_train, f)
        with open(os.path.join(self.pickle_path,'X_test.pickle'), 'wb') as f1:
            pickle.dump(self.X_test, f1)
        with open(os.path.join(self.pickle_path,'y_train.pickle'), 'wb') as f2:
            pickle.dump(self.y_train, f2)
        with open(os.path.join(self.pickle_path,'y_test.pickle'), 'wb') as f3:
            pickle.dump(self.y_test, f3)
        with open(os.path.join(self.pickle_path,'X_dev.pickle'), 'wb') as f4:
            pickle.dump(self.X_dev, f4)
    def load_pickle(self):
        with open(os.path.join(self.pickle_path,'X_train.pickle'), 'rb') as f:
            self.X_train=pickle.load(f)
        with open(os.path.join(self.pickle_path,'X_test.pickle'), 'rb') as f1:
            self.X_test=pickle.load(f1)
        with open(os.path.join(self.pickle_path,'y_train.pickle'), 'rb') as f2:
            self.y_train=pickle.load(f2)
        with open(os.path.join(self.pickle_path,'y_test.pickle'), 'rb') as f3:
            self.y_test=pickle.load(f3)
        with open(os.path.join(self.pickle_path,'X_dev.pickle'), 'rb') as f4:
            self.X_dev = pickle.load(f4)


        






