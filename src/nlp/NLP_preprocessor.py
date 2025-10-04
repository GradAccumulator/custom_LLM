import torch
from src.utils.dev_utils import reading, writing, parent_timer, timer, print_log
from winsound import Beep
from src.utils.processing.data_processing import batchify
import src.utils.dev_utils as dev_utils
import src.utils.processing.data_processing as dp
from os.path import join

class NLP_Preprocessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.load_preprocessed = config.load_preprocessed_dataset_file
        if self.load_preprocessed:
            self.dataset_path = config.PREPROCESSED_DATASET_DIR
        else:
            self.dataset_path = config.RAW_DATA_PATH
        self.train_x = None
        self.train_label = None
        self.val_x = None
        self.val_label = None
    
    def __call__(self):
        return self.preprocess()
    def preprocess(self):
        dev_utils.print_log("데이터 전처리 시작")
        if not self.load_preprocessed:
            with dev_utils.timer("raw 데이터셋 읽는 시간"):
                trainset, valset = self._read_raw_datasets()

            with dev_utils.timer("trainset 섞는시간"):
                trainset = dp.shuffle(trainset)
            with dev_utils.timer("valset 섞는 시간"):
                valset = dp.shuffle(valset)

            with dev_utils.parent_timer("trainset 인코딩& 패딩"):
                self.train_x,self.train_label = self._encode_and_pad(trainset, self.config.goal_trainset_len*1.6)
            with dev_utils.parent_timer("valset 인코딩&패딩"):
                self.val_x,self.val_label     = self._encode_and_pad(valset, self.config.goal_trainset_len*1.6)

            with dev_utils.timer("전처리된 데이터셋 저장"):
                torch.save(self.train_x, join(self.config.PREPROCESSED_DATASET_DIR, "train_x.pth"))
                torch.save(self.train_label, join(self.config.PREPROCESSED_DATASET_DIR, "train_label.pth"))
                torch.save(self.val_x, join(self.config.PREPROCESSED_DATASET_DIR, "val_x.pth"))
                torch.save(self.val_label, join(self.config.PREPROCESSED_DATASET_DIR, "val_label.pth"))
        else:
            with dev_utils.timer("전처리된 데이터셋 읽는 시간"):
                self.train_x = torch.load(join(self.dataset_path, "train_x.pth"))
                self.train_label = torch.load(join(self.dataset_path, "train_label.pth"))
                self.val_x = torch.load(join(self.dataset_path, "val_x.pth"))
                self.val_label = torch.load(join(self.dataset_path, "val_label.pth"))
        
        dp.check_and_raise(len(self.train_x)<self.config.goal_trainset_len, 
            f"[Preprocessor] 예상 trainset 길이와 실제 trainset 길이가 다릅니다. 예상:{self.config.goal_trainset_len}, 실제:{len(self.train_x)}")
        dp.check_and_raise(len(self.val_x)<self.config.goal_valset_len,
            f"[Preprocessor] 예상 valset 길이와 실제 valset 길이가 다릅니다. 예상:{self.config.goal_valset_len}, 실제:{len(self.val_x)}")
        
        with dev_utils.timer("trainset 슬라이싱"):
            self.train_x = self.train_x[:self.config.goal_trainset_len]
            self.train_label = self.train_label[:self.config.goal_trainset_len]

        with dev_utils.timer("valset 슬라이싱"):
            self.val_x = self.val_x[:self.config.goal_valset_len]
            self.val_label = self.val_label[:self.config.goal_valset_len]

        with dev_utils.timer("batchify"):
            self._batchify()

        dev_utils.print_log("데이터 전처리 완료")
        return (self.train_x, self.train_label, self.val_x, self.val_label)


    def _read_raw_datasets(self):
        trainset_len = 0
        valset_len = 0  
        trainset_strlen = 0
        valset_strlen = 0
        valset = []
        trainset = []
        with dev_utils.reading(self.dataset_path) as f:
            for line in f:
                line = line.strip()
                if valset_len<self.config.goal_valset_len and \
                    (valset_strlen/(self.config.goal_valset_len*self.model.seq_len)) < 5:
                    valset.append(line)
                    valset_len+=1
                    valset_strlen += len(line)
                elif trainset_len<self.config.goal_trainset_len and \
                    (trainset_strlen/(self.config.goal_trainset_len*self.model.seq_len))<50:
                    trainset.append(line)
                    trainset_len += 1
                    trainset_strlen+=len(line)
                else:
                    break
        return (trainset, valset)
        

    def _encode_and_pad(self, dataset, max_len):
        
        with parent_timer("인코딩&패딩"):
            batchified:list[list] = batchify(dataset, batch_size=100_000)
            inputs, labels = [],[]
            length = len(batchified)
            print("total length :", length)
            for idx,batch in enumerate(batchified, start=1):
                i, l = self.model.encode(batch, is_dataset=True,max_length=500000)
                inputs.append(i)
                labels.append(l)
                del batch
                print(f"{idx/length*100:.2f}% complete...")
            inputs = torch.cat(inputs, dim=0)
            labels = torch.cat(labels, dim=0)
            inputs,labels = dp.shuffle_dataset(inputs, labels)
        return (inputs, labels)

    def _batchify(self):
        self.train_x, self.train_label = dp.batchify(self.train_x, self.train_label,batch_size=self.config.batch_size)
        self.val_x, self.val_label = dp.batchify(self.val_x, self.val_label, batch_size=self.config.val_batch_size)