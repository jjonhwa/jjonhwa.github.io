---
layout: post
title: "Word2Vec"
categories: booststudy
tags: plus
comments: true
---
Word2Vec에 대한 요약과 구현을 진행한다. Word2Vec에 대한 보다 자세한 설명은 [여기](https://jjonhwa.github.io/boostcamp/2021/09/06/boostcamp-main-Ustage_Day_25-2/#1-word-embedding)에서 확인할 수 있다.

## 목차
- [1. word2vec이란?](#1-word2vec이란)
- [2. word2vec 요약 설명](#2-word2vec-요약-설명)
- [3. word2vec 구현](#3-word2vec-구현)

## 1. word2vec이란?
[위키백과 - 워드투벡터](https://wikidocs.net/22660)에서 참고하였습니다.

- 단어 간의 유사도를 반영하기 위해 단어의 의미를 벡터화 시킬 때 사용하는 방법 중 하나이다.
- [한국어 유사도 연산](http://w.elnn.kr/search/)를 통해 한국어 단어 간의 유사도 연산을 할 수 있고, 실제로는 word2vec 벡터를 바탕으로 단어들 사이의 연산이 진행되는 것이다. 예를 들어, '한국 - 서울 + 도쿄'를 연산하면 '일본'이라는 결과가 나타나게 된다.

## 2. word2vec 요약 설명
- one-hot 인코딩으로는 단어 간의 유사도를 계산할 수 없기 때문에, 단어 간의 유사도를 측정할 수 있도록 단어의 의미를 여러 차원에 분산하여 표현하는 분산 표현을 사용한다.
- '비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다'라는 아이디어를 바탕으로 등장하였다.
- word2vec은 각 단어가 주변에 등장하는 단어들과 그 의미가 비슷하다는 사실을 착안하고, 이에 따라 주어진 학습데이터를 바탕으로 target word 주변에 나타나는 단어들의 확률 분포를 예측한다.
- word2vec에는 CBOW와 Skip-Gram 두 가지의 학습 방법이 있다.
    - CBOW(Contiuous Bag of Words) : 주변 단어를 입력으로 주고, target word를 예측하는 방식이다.
    - Skip-Gram : word를 입력으로 주고, 주변 단어를 숨긴 채 이를 예측하는 방식이다.
- 보다 자세한 설명은 [여기](https://jjonhwa.github.io/boostcamp/2021/09/06/boostcamp-main-Ustage_Day_25-2/#1-word-embedding)에서 확인할 수 있다.

## 3. word2vec 구현
### 3-1. 패키지 import
먼저, Terminal 환경에서 jdk를 설치해준다. konlpy의 Twitter(OKT) tokenizer를 사용하는데 있어서, jdk를 필요로하므로 이를 설치해준다. 설치가 되어있다면 넘어가도록 하자.
``` 
$sudo pat install default-jdk 
```
필요로 하는 패키지를 설치 및 불러온다.
```
!pip install konlpy # 최초 한 번만 진행해준다.

from tqdm import tdqm
from konlpy.tag import Okt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collection import defaultdict

import torch
```

### 3-2. 데이터 전처리
#### 학습 및 테스트 데이터 입력

```
train_data = [
  "진짜 맛있어요. 강추강추",
  "안가면 손해, 맛집이에요.",
  "리뷰보고 갔는데 생각했던거와 다르네요.",
  "최고에요! 다음에 또 오겠습니다.",
  "주방 위생이 별로인거 같아요. 신경좀 쓰셔야 할듯..",
  "중요한 날에 방문했는데 서비스가 좋아서 만족했습니다! 맛도 괜찮아요",
  "생각보다 비싸요.. 가성비는 별로네요",
  "재방문의사 100프로!!",
  "서비스가 좋아요~ 친절하시네요!",
  "청소는 하시나요... 매장이 조금 더럽네요,,",
  "두번 다시 안갈듯.. 별로별로"
  "음식도 형편없고, 가격도 비싸요.",
  "음식 맛있어요! 가성비 괜찮은듯!"]

test_words = ['음식', '맛', '서비스', '위생', '가격']
```
- train data를 바탕으로 단어간의 유사도를 구하고, test word에 대한 벡터값을 찾는다.
- 이렇게 찾아진 word의 vector값을 바탕으로 단어간의 유사도를 측정할 수 있다.

#### Tokenization
- tokenization을 통해, "안가면 손해, 맛집이에요." -> ['안', '가면', '손해', ',', '맛집', '이에요', '.']와 같이 만들어준다.

```
tokenizer = Okt() # Twitter(Okt) tokenizer를 사용하여 토큰화 진행

def make_tokenized(data) :
    tokenized = []
    for sent in tqdm(data) :
        tokens = tokenizer.morphs(sent, stem = True) # 형태소 단위로 문장 토큰화하고, stem = True로 줌으로서, 각 단어의 어간을 추출한다.
        tokenized.append(tokens)
    return tokenized

train_tokenized = make_tokenized(train_data)
```

#### Vocab 생성
- 가장 많이 등장한 토큰 순서대로 나열해주고 이를 index 매겨 vocab을 생성한다.
- vocab을 생성하는 방법은 다양하게 존재한다.
    
```
word_count = defaultdict(int) # 가장 많이 등장한 token 순서대로 dict형태로 나열한다.
for tokens in tqdm(train_tokenized) :
    for token in tokens :
        word_count[token] += 1
word_count = sorted(word_count.items(), key = lambda x: x[1], reverse = True)

word2idx = {} # 가장 많이 등장한 token 순서대로 index를 주어 Vocab 생성
for word, count in tqdm(word_count) :
    if word not in word2idx :
        word2idx[word] = len(word2idx)
```

### 3-3. Dataset class 정의
#### CBOW Dataset
```
class CBOWDataset(Dataset) :
    def __init__(self, train_tokenized, vocab, window_size = 2) :
        self.x = []
        self.y = []
        
        for tokens in tqdm(train_tokenized) :
            token_ids = [vocab[token] for token in tokens] # word to index
            for i, id in enumerate(token_ids) : # CBOW 예측 방법 적용. 주변 단어를 바탕으로 중심 단어 예측
                if i-window_size >= 0 and i+window_size < len(token_ids) :
                    self.x.append(token_ids[i-window_size:i] + token_ids[i+1 : i+window_size+1]) # i번째 단어에 대한 주변 단어의 인덱스를 x에 추가
                    self.y.append(id) # i번째 단어에 대한 index를 y에 추가
        
        self.x = torch.LongTensor(self.x) # 64bit integer type으로 변환
        self.y = torch.LongTensor(self.y)
    
    def __len__(self) :
        return self.x.shape[0]
    
    def __getitem__(self, idx) :
        return self.x[idx], self.y[idx]
```

#### SkipGram Dataset
```
class SkipGramDataset(Dataset) :
    def __init__(self, train_tokenized, vocab, window_size = 2) :
        self.x = []
        self.y = []
        
        for tokens in tqdm(train_tokenized) :
            token_ids = [vocab[token] for token in tokens]
            for i, id in enumerate(token_ids) : # SkipGram 예측 방법 적용. 중심 단어를 바탕으로 주변 단어 예측
                if i-window_size >= 0 and i+window_size < len(token_ids) :
                    self.y += (token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1]) # 주변 단어를 y로 입력
                    self.x += [id] * 2 * window_size # 중심 단어를 주변 단어의 개수에 맞게 늘려주어 x에 입력
        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)
    
    def __len__(self) :
        return self.x.shape[0]
        
    def __getitem__(self, idx) :
        return self.x[idx], self.y[idx]
```

```
cbow_set = CBOWDataset(train_tokenized, word2idx)
skipgram_set = SkipGramDataset(train_tokenized, word2idx)
```

### 3-3. 모델 구현 및 DataLoader
- `self.embedding` : vocab_size 크기의 one-hot vector를 유사도를 구하기 위한 분산 표현을 나타내기 위해 특정 크기의 dimension으로 embedding하는 layer
- `self.linear` : 변환된 embedding vector를 원래 vocab_size로 바꾸는 layer

#### CBOW model class
```
class CBOW(nn.Module) :
    def __init__(self, vocab_size, dim) :
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim, sparse = True) # sparse = True를 이용해, matrix에서 0인 부분을 제외하고 계산한다.
        self.linear = nn.Linear(dim, vocab_size)
        
    def forward(self, x) : 
        embeddings = self.embedding(x) # x를 one-hot vector가 아닌 embedding vector로 변환한다.
        embeddings = torch.sum(embeddings, dim = 1) # window에 있는 단어들에 대한 값을 합친다.
        output = self.linear(embeddings)
        return output
```

#### SkipGram model class
```
class SkipGram(nn.Module) :
    def __init__(self, vocab_size, dim) :
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim, sparse = True)
        self.linear = nn.Linear(dim, vocab_size)
    
    def forward(self, x) :
        embeddings = self.embedding(x) # CBOW와 달리 window size가 없다.
        output = self.linear(embeddings)
        return output
```

#### 모델 입력 및 DataLoader 입력
```
cbow = CBOW(vocab_size = len(word2idx), dim = 256)
skipgram = SkipGram(vocab_size = len(word2idx), dim = 256)

batch_size = 4
cbow_loader = DataLoader(cbow_set, batch_size = batch_size)
skipgram_loader = DataLoader(skipgram_set, batch_size = batch_size)
```

### 3-4. 모델 학습 및 테스트
#### 모델 학습
```
learning_rate = 5e-4
num_epochs = 5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

method = 'cbow' # 'cbow' or 'skip_gram'을 입력 
if method == 'cbow' :
    model = cbow
    loader = cbow_loader
elif method == 'skip_gram' :
    model = skipgram
    loader = skipgram_loader
else :
    raise ValueError ("You have to insert one of 'cbow' or 'skip_gram'")

model.train()
model = model.to(device)
optim = torch.optim.SGD(cbow.parameters(), lr = learning_rate)
loss_function = nn.CrossEntropyLoss()

for e in range(1, num_epochs + 1) :
    print(f"Epoch: {e}")
    for batch in tqdm(loader) :
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(x)
        
        optim.zero_grad()
        loss = loss_function(output, y)
        loss.backward()
        optim.step()
    
    print(f"Train loss per epoch: {loss.item()}")
print('Finished')
```

#### 테스트
```
for word in test_words :
    input_id = torch.LongTensor([word2idx[word]]).to(device)
    emb = model.embedding(input_id)
    
    print(f"Word: {word}")
    print(max(emb.squeeze(0)))
```

- SKipGram의 경우 다음과 같이 출력된다.

Word: 음식  
tensor(2.7496, device='cuda:0', grad_fn=<UnbindBackward>)  
Word: 맛  
tensor(2.7587, device='cuda:0', grad_fn=<UnbindBackward>)  
Word: 서비스  
tensor(2.7119, device='cuda:0', grad_fn=<UnbindBackward>)  
Word: 위생  
tensor(3.1804, device='cuda:0', grad_fn=<UnbindBackward>)  
Word: 가격  
tensor(2.8926, device='cuda:0', grad_fn=<UnbindBackward>)  
