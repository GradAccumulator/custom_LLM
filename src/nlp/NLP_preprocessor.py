import torch
from src.utils.dev_utils import reading, writing, parent_timer, timer, print_log
from winsound import Beep
from src.utils.processing.data_processing import batchify

class NLP_Preprocessor:
    def __init__(self, model, config, dev_utils, dp):
        self.model = model
        self.config = config
        self.load_preprocessed = config.load_preprocessed_dataset_file
        if self.load_preprocessed:
            self.dataset_path = config.PREPROCESSED_DATASET_PATH
        else:
            self.dataset_path = config.RAW_DATA_PATH
        self.utils = dev_utils
        self.dp = dp
        self.train_x = None
        self.train_label = None
        self.val_x = None
        self.val_label = None

    def raw_data_reprocess(self):
        import re
        front_dots = re.compile(r'^\.{2,}')
        behind_dots = re.compile(r'\.{2,}$')
        middle_dots = re.compile(r'(?<=\S)\.{2,}(?=\S)')
        many_spaces = re.compile(r'\s+')
        ㅋㅋㅋㅋ = re.compile(r"\ㅋ{3,}")
        한자 = re.compile(r'[\u4E00-\u9FFF]+')
        i =0
        with parent_timer("데이터 다시 전처리"):
            with reading(self.config.RAW_DATA_PATH) as f:
                texts = []
                for line in f:
                    line = line.strip()
                    line = front_dots.sub('…', line)
                    line = behind_dots.sub('…', line)
                    line = middle_dots.sub(' ', line)
                    line = 한자.sub('', line)
                    line = ㅋㅋㅋㅋ.sub('ㅋㅋㅋ', line)
                    line = line.replace('○', '<SPEAKER>')
                    line = line.replace('■', '')
                    line = line.replace('「', "'")
                    line = line.replace('」', "'")
                    line = line.replace('\\', '')
                    line = line.replace('-', '')
                    line = line.replace('π', '파이')
                    line = line.replace('�', '')
                    line = line.replace(' . ', ',')
                    line = line.replace('·', ',')
                    line = line.replace('▷', '')
                    line = line.replace('Slave', '노예')
                    line = line.replace('COMPUTER', '컴퓨터')
                    line = many_spaces.sub(' ', line)
                    
                    texts.append(line)
                    if (i:=i+1)%1000==0:
                        print_log(f"{i}번째 데이터 재처리 완료")
        with timer("데이터 저장"):
            with writing(self.config.RAW_DATA_PATH) as f:
                f.writelines([f"{line}\n" for line in texts])
        Beep(300, 100)
        Beep(300, 100)
        Beep(300, 100)
    
    def __call__(self):
        return self.preprocess()
    def preprocess(self):
        self.utils.print_log("데이터 전처리 시작")
        if not self.load_preprocessed:
            with self.utils.timer("raw 데이터셋 읽는 시간"):
                trainset, valset = self._read_raw_datasets()

            with self.utils.timer("trainset 섞는시간"):
                trainset = self.dp.shuffle(trainset)
            with self.utils.timer("valset 섞는 시간"):
                valset = self.dp.shuffle(valset)

            with self.utils.parent_timer("trainset 인코딩& 패딩"):
                self.train_x,self.train_label = self._encode_and_pad(trainset, self.config.goal_trainset_len*1.6)
            with self.utils.parent_timer("valset 인코딩&패딩"):
                self.val_x,self.val_label     = self._encode_and_pad(valset, self.config.goal_trainset_len*1.6)

            with self.utils.timer("전처리된 데이터셋 저장"):
                D = {
                    "train_x":self.train_x,
                    "train_label":self.train_label,
                    "val_x":self.val_x,
                    "val_label":self.val_label,
                }
                torch.save(D, self.config.PREPROCESSED_DATASET_PATH)
        else:
            with self.utils.timer("전처리된 데이터셋 읽는 시간"):
                D = torch.load(self.dataset_path)
                self.train_x       = D['train_x']
                self.train_label   = D['train_label']
                self.val_x      = D['val_x']
                self.val_label      = D['val_label']
        
        self.dp.check_and_raise(len(self.train_x)<self.config.goal_trainset_len, 
            f"[Preprocessor] 예상 trainset 길이와 실제 trainset 길이가 다릅니다. 예상:{self.config.goal_trainset_len}, 실제:{len(self.train_x)}")
        self.dp.check_and_raise(len(self.val_x)<self.config.goal_valset_len,
            f"[Preprocessor] 예상 valset 길이와 실제 valset 길이가 다릅니다. 예상:{self.config.goal_valset_len}, 실제:{len(self.val_x)}")
        
        with self.utils.timer("trainset 슬라이싱"):
            self.train_x = self.train_x[:self.config.goal_trainset_len]
            self.train_label = self.train_label[:self.config.goal_trainset_len]

        with self.utils.timer("valset 슬라이싱"):
            self.val_x = self.val_x[:self.config.goal_valset_len]
            self.val_label = self.val_label[:self.config.goal_valset_len]

        with self.utils.timer("batchify"):
            self._batchify()

        self.utils.print_log("데이터 전처리 완료")
        return (self.train_x, self.train_label, self.val_x, self.val_label)


    def _read_raw_datasets(self):
        trainset_len = 0
        valset_len = 0
        trainset_strlen = 0
        valset_strlen = 0
        valset = []
        trainset = []
        with self.utils.reading(self.dataset_path) as f:
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
            batchified = batchify(dataset, batch_size=100000)
            inputs, labels = [],[]
            for batch in batchified:
                i, l = self.model.encode(batch, is_dataset=True,max_length=500000)
                inputs.append(i)
                labels.append(l)
            inputs = torch.cat(inputs, dim=0)
            labels = torch.cat(labels, dim=0)
            inputs,labels = self.dp.shuffle_dataset(inputs, labels)
        return (inputs, labels)

    def _batchify(self):
        self.train_x, self.train_label = self.dp.batchify(self.train_x, self.train_label,batch_size=self.config.batch_size)
        self.val_x, self.val_label = self.dp.batchify(self.val_x, self.val_label, batch_size=self.config.val_batch_size)