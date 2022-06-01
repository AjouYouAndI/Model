import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import numpy as np
import gluonnlp as nlp
from mxnet import gluon

from BERTDataset import BERTDataset

from fastapi import APIRouter

PATH = '/home/ec2-user/YouAndI/app/weights/'

router = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"description": "Not Found"}},
)


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

url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
f = gluon.utils.download(url)
vocab = torch.load(PATH+'vocab_obj.pth')

tok = nlp.data.BERTSPTokenizer(f, vocab, lower=True)

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
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")

        return(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

@router.put("/predict")
async def prdecit_result():
    input = "아이고 좋아라"
    result = await predict(input)
    return result