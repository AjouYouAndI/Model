PATH = '/home/ec2-user/YouAndI/app/weights/'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import numpy as np
import gluonnlp as nlp
from mxnet import gluon

from flask import Flask, jsonify, request


app = Flask(__name__)
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    
# BERT 모델에 들어가기 위한 dataset을 만들어주는 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(PATH + 'model.pt',map_location=device)  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt', map_location=device))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load(PATH + 'all.tar', map_location=device)   # dict 불러오기 - 매개변수 값들이 담겨있는 state_dict 객체.
model.load_state_dict(checkpoint['model'])
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
optimizer.load_state_dict(checkpoint['optimizer'])


def predict(predict_sentence): #예측 함수 구현

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, 64, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=64, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람")
            elif np.argmax(logits) == 2:
                test_eval.append("분노")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔")
            elif np.argmax(logits) == 4:
                test_eval.append("중립")
            elif np.argmax(logits) == 5:
                test_eval.append("행복")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오")

        return test_eval[0]

url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
f = gluon.utils.download(url)
vocab = torch.load(PATH+'vocab_obj.pth')

tok = nlp.data.BERTSPTokenizer(f, vocab, lower=True)

@app.route('/')
def hello():
    return "hello"

@app.route('/predict')
def predict_read():
    sentence = request.args.get('string')
    return str(predict(sentence))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)







    
