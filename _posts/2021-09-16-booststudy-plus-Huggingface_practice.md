---
layout: post
title: "Hugging Face Practice"
categories: booststudy
tags: plus
comments: true
---
hugging face를 사용하는 간단한 tutorial 느낌의 practice를 한다.

hugging face의 tutorial은 [hugging face tutorial](https://huggingface.co/course/chapter1?fw=pt)에서 확인할 수 있으며 본 과정은 boostcamp 실습을 바탕으로 작성했습니다.

## 목차
- [Import Module](#import-module)
- [1. 모델 불러오기](#1-모델-불러오기)
- [2. Tokenizer](#2-tokenizer)
- [3. Data Preprocess](#3-data-preprocess)
- [4. BERT 모델 사용 및 응용](#4-bert-모델-사용-및-응용)

## Import Module
```
!pip install transformers # 최조 1회에만 설치한다.
```

```
from torch import nn
from tqdm import tqdm

import torch
import transformers
```

## 1. 모델 불러오기
모델은 [huggind face - model](https://huggingface.co/models)에서 사용하고자 하는 모델의 이름을 가져와 아래의 name에 넣어서 사용할 수 있다.
```
bert_name = 'bert-base-uncased'
```

아래의 코드와 같이, 모델에서 사용하는 configuration과 tokenizer, model을 다음의 함수를 사용하여 불러올 수 있다.
```
config = transformers.BertConfig.from_pretrained(bert_name)
tokenizer = transformers.BertTokenizer.from_pretrained(bert_name)
model = transformers.BertModel.from_pretraind(bert_name)
```

## 2. Tokenizer
위에서 불러온 tokenizer를 사용하여 주어진 문장을 token화 해본다.

### 2-1. tokenizer 직접 적용
```
sentence = 'I want to go home.'
output = tokenizer(sentence)
```
위의 코드처럼 output을 주게 되면 아래의 출려곽 같이, input_ids, token_type_ids, attention_mask를 출력해주는데 이는 model의 입력에 필요한 세가지 입력값이며 이들은 각각 다음과 같은 의미를 지닌다.
- input_ids : input token으로서 입력 단어를 토큰화한 값들의 list형태이며 맨 앞과 맨 뒤에 sos token과 eos token이 붙는 형태이다.
- token_type_ids : segment에 대한 정보로, 여러 입력 문장이 있을 때 [SEP] token을 기준으로 같은 문장은 전부 같은 id로 채워넣는 형태이다.
- attention_mask : 패딩인 토큰은 0, 패딩이 아닌 토큰은 1로 채워주는 형태이다.

```
'input_ids': [101, 1045, 2215, 2000, 2175, 2188, 1012, 102]
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0]
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

### 2-2. tokenizer의 함수로 적용
- tokenize 함수
    - 주어진 문장을 token화 시킨다.   
    
```
tokenized = tokenizer.tokenize(sentence)
print(tokenized)
```
```
['i', 'want', 'to', 'go', 'home', '.']
```

- get_vocab 함수
    - vocabulary를 확인한다. 
    
```
vocab = tokenizer.get_vocab()

print(list(vocab.items())[:5]) # vocab을 보기 좋게 출력하기 위하여 list화 시켰다.
print(len(vocab)
```
```
[('[PAD]', 0), ('[unused0]', 1), ('[unused1]', 2), ('[unused2]', 3), ('[unused3]', 4)]
30522
```
```
print(vocab['[CLS]'])
print(vocab['[SEP]'])
```
```
101
102
```

- token_ids 출력 방법
    - [vocab[token] for token in tokenized] : 직접적인 vocab을 활용한 word to idx
    - [tokenizer._convert_token_to_id(token) for token in tokenized] : tokenizer의 함수를 활용하여 각 단어별로 idx를 바꾸는 방식의 word to idx, 앞서 vocab을 활용한 방식과 결과가 같다.
    - tokenizer.convert_tokens_to_ids(tokenized) : 문장 전체를 바로 word to idx 변환, 앞서 vocab을 활용한 방식과 결과가 같다.
    - tokenizer.encode(sentence) : 토큰화되지 않는 word 자체로 이루어진 sentence를 입력으로 주고 이를 tokenizer의 encode함수를 통해 word to idx를 바로 진행해준다. 더불어, encode에서는 <sos>, <eos> token을 추가해서 출력해준다. BERT의 경우 맨 처음에는 [CLS] token이, 맨 마지막에는 [SEP] token이 붙는다.

```
token_ids_1 = [vocab[token] for token in tokenized]
token_ids_2 = [tokenizer._convert_token_to_id(token) for token in tokenized]
token_ids_3 = tokenizer.convert_tokens_to_ids(tokenized)
token_ids_4 = tokenizer.encode(sentence)

print(token_ids_1)
print(token_ids_2)
print(token_ids_3)
print(token_ids_4)
```
```
[1045, 2215, 2000, 2175, 2188, 1012]
[1045, 2215, 2000, 2175, 2188, 1012]
[1045, 2215, 2000, 2175, 2188, 1012]
[101, 1045, 2215, 2000, 2175, 2188, 1012, 102]
```

- token을 원문으로 되돌리기
    - tokenizer.convert_tokens_to_string(tokenized) : tokenized된 단어 즉, 토큰화된 문장을 원래의 하나의 sentence로 변환해준다.
    - tokenizer.convert_ids_to_tokens(token_ids) : token id로부터 token word로 변환해주며, 이 때, 맨 앞, 맨 뒤에는 앞서 설명했던 [CLS], [SEP] token이 붙는다.
    - tokenizer.convert_tokens_to_string(tokens) : tokens를 입력으로 주어 convert_tokens_to_string 함수를 거치게되면 앞의 하나의 sentence로 변했던 것처럼 special token과 함께 token들을 하나의 sentence로 만들어준다.

```
sentence = tokenizer.convert_tokens_to_string(tokenized)
tokens = tokenizer.convert_ids_to_tokens(token_ids)
sentence = tokenizer.convert_tokens_to_string(tokens)

print(sentence)
print(tokens)
print(sentence)
```
```
i want to go home .
['[CLS]', 'i', 'want', 'to', 'go', 'home', '.', '[SEP]']
[CLS] i want to go home . [SEP]
```

- 두 개 이상의 문장 입력
    - input_ids : 문장 사이에 자동으로 [SEP] token을 추가해준다.
    - token_type_ids : 서로 다른 문장은 다른 idx 부여
    - attention_mask : 위에서의 설명과 동일하다.

```
print(tokenizer('I want to go home.', 'Me too.')
```
```
'input_ids': [101, 1045, 2215, 2000, 2175, 2188, 1012, 102, 2033, 2205, 1012, 102]
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## 3. Data Preprocess
- model에 주입할 수 있는 형태로 데이터를 전처리하고 이는 model의 input_ids로 들어가는 입력값이 된다.

```
data = [
  "I want to go home.",
  "My dog's name is Max.",
  "Natural Language Processing is my favorite research field.",
  "Welcome. How can I help you?",
  "Shoot for the moon. Even if you miss, you'll land among the stars."
]

max_len = 0 # token_ids의 길이를 max_len으로 맞춰준다.
batch = []
for sent in tqdm(data) :
    token_ids = tokenizer.encode(sent)    # sentence to idx
    max_len = max(max_len, len(token_ids) # 가장 긴 문장의 길이로 max_len을 입력한다.
    batch.append(token_ids)               # list에 담아주기

pad_id = tokenizer._convert_token_to_id('[PAD]') # [PAD] token을 id화
for i, token_ids in enumerate(tqdm(batch)) :
    if len(token_ids) < max_len : # token의 길이가 max_len보다 작을 경우 [PAD] 토큰을 붙여 길이를 맞춰준다.
        batch[i] = token_ids + [pad_id] * (max_len - len(token_ids))

batch = torch.LongTensor(batch) # type을 64bit integer(signed)로 batch를 바꾼다.

print(batch)
print(batch.shape)
```
```
tensor([[ 101, 1045, 2215, 2000, 2175, 2188, 1012,  102,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0],
        [ 101, 2026, 3899, 1005, 1055, 2171, 2003, 4098, 1012,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0],
        [ 101, 3019, 2653, 6364, 2003, 2026, 5440, 2470, 2492, 1012,  102,    0,
            0,    0,    0,    0,    0,    0,    0,    0],
        [ 101, 6160, 1012, 2129, 2064, 1045, 2393, 2017, 1029,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0],
        [ 101, 5607, 2005, 1996, 4231, 1012, 2130, 2065, 2017, 3335, 1010, 2017,
         1005, 2222, 2455, 2426, 1996, 3340, 1012,  102]])
torch.Size([5, 20])
```

- batch_mask tensor를 만들어 attentoin_mask의 입력값으로 넣어준다.
- 이는 pad token일 경우 0, 아닐 경우 1을 가지는 형태이다.
                              
```
batch_mask = (batch != pad_id).float() # batch가 pad일경우 False, 아닐 경우 true를 주고 이를 float로 변경하여 0과 1로 치환한다.

print(batch_mask)
print(batch_shape) # shape은 (문장의 수 x token의 max_length) 이다.
```
```
tensor([[1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1.]])
torch.Size([5, 20])
```

## 4. BERT 모델 사용 및 응용
- last hidden state & pooler output 확인
    - BERT model은 최종 출력으로 last hidden state와 pooler를 내놓는다.
    - last hidden state는 word의 encoding vector로서 Batch_size x Max_length x Hidden_dimension_size의 차원을 가진다.
    - pooler는 downstream task에 적용하기 위한 [CLS] token에 대한 embedding 값으로 [CLS] token에 대한 hidden state vector를 추가적으로 linear layer and Tanh activation fuction을 적용한 값이다. shape은 Batch_size x Hidden_dimension_size를 가지게 된다. ([stack overflow 참고](https://stackoverflow.com/questions/63377198/what-is-the-difference-between-pulled-output-and-sequence-output-in-bert-layer))

```
outputs = model(input_ids = batch, attention_mask = batch_mask)
last_hidden_states = outputs[0]
pooler_output = outputs[1]

print(last_hidden_states.shape)
print(pooler_output.shape)
```
```
torch.Size([5, 20, 768])
torch.Size([5, 768])
```

### 4-1. Sentence-level Classification
- BERT 모델에서 Sentence-level classification을 진행하기 위해서는 [CLS] token을 사용한다.
- BERT 모델에서 최종적인 출력인 last_hidden_states의 token에서의 0번째 vector가 [CLS] hidden state vector이므로 이 값을 최종 fully connected layer를 거쳐 classification하고자 하는 개수로 출력해준다.

```
cls_output = last_hidden_states[:, 0, :] # token의 0번째 vector가 cls output hidden state vector이다.

num_classes = 10 # 10개의 class에 대한 classification을 진행한다고 가정하자.
sent_linear = nn.Linear(config.hidden_size, num_claases) # 768 to 10인 fully connected layer를 추가한다.
sent_output = sent_linear(cls_output) # cls_output을 최종적으로 fully connected layer를 통과해 최종 예측을 진행할 수 있다.
```

### 4-2. 그 외에 다양한 head를 추가한 모델
- Sequence-level의 classification : BertForSequenceClassification.from_pretrained(bert_name)
- MLM : BertForMaskedLM.from_pretrained(bert_name, config = config)
- 이 외에서 더욱 다양한 모델들이 존재한다. [추가 모델 찾아보기](https://huggingface.co/transformers/model_doc/bert.html)에서 확인하도록 하자.

Sequence-level의 classification의 경우, 마지막 layer가 hidden_size to 2로 구성되어 있고, MLM의 경우 hidden_size to vacab_size로 구성되어 vocab 중에서 어떤 단어를 출력할 것인지 예측하도록 한다.
