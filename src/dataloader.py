import os

class DataLoader:
    """docstring for ClassName"""
    def __init__(self):
        self.data_root='../input_data'
        self.ok_train_path = os.path.join(self.data_root,'train/OK')
        self.ng_train_path = os.path.join(self.data_root,'train/OK')
        self.ok_test_path = os.path.join(self.data_root,'train/OK')
        self.ng_test_path = os.path.join(self.data_root,'train/OK')
        self.dev_path = os.path.join(self.data_root,'dev')
    def data_num(self):
        def get_num(path):
            return len(os.listdir(path))
        print('trainOK',get_num(self.ok_train_path),'件')
        print('trainNG',get_num(self.ng_train_path),'件')
        print('testOK', get_num(self.ok_test_path),'件')
        print('testNG', get_num(self.ng_test_path),'件')
        print('dev',    get_num(self.dev_path),'件')
        