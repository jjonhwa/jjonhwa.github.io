---
layout: post
title: "Huggingface & CLS"
categories: booststudy
tags: plus
comments: true
---
Huggingface에 대한 다양한 실습에 대하여 이야기하며, cls 토큰에 대하여 가볍게 이야기한다.

## 목차
- [1. Huggingface](#1-huggingface)
- [2. CLS 토큰은 문장을 대표할까?](#2-cls-토큰은-문장을-대표할까)

## 1. Huggingface
NLP 모델은 간단하게 다음과 같은 Pipeline으로 구성된다.
input -> Tokenization -> Model training -> Inference -> Post-Processing(task dependent) -> Output

### 1-1. Tokenizer
- Text data를 모델이 이해할 수 있는 형태로 변환하기 위해 Tokenization을 진행한다.
- Text data를 token화하고 특정 숫자로 encoding하는 과정을 모두 수행하는 것이 transformers tokenizer의 역할이다.
- 다음의 코드처럼 활용할 수 있다.
```
from transformers import AutoTokenizer

example = '나는 정말 열심히 노력하고 있다.' # 간단한 예제를 가져온다.
model_name = 'bert-base-case' # huggingface의 다양한 모델을 이름만 가져와서 사용할 수 있다.
tokenizer = AutoTokenizer.from_pretrained(model_name)

```
#### Tokenizer 사용시 주의사항. (중요)
- **Tokenizer가 내가 사용하는 데이터의 언어를 이해할 수 있는가?**
    - 내가 사용하는 언어의 tokenizer를 사용하지 않으면 대부분의 token이 unknown token으로 encoding되어 학습이 무의미해진다. 
- **사용하고자 하는 pre-trained model과 동일한 tokenizer를 사용하고 있는가?**
    - vocab_size error가 발생할 수 있다.
    - special token이 unknown token으로 바뀔 수 있다.
- 서로 다른 모델에서 단어의 개수, special token이 완전히 일치하는 것은 우연의 일치일 뿐이다. 이럴 경우 사용할 수도 있지만 올바른 방법은 아니다.

### 1-2. Config
- 사전 학습 모델을 사용하기 위해서는 사전학습 모델이 가진 setting을 그대로 가져와서 사용해야한다.
- 모델마다 vocab_size, hidden_dim등의 파라미터들이 다르므로 huggingface transformers는 이를 가져오는 Config를 제공한다.
- 다음의 코드를 활용하여 config를 불러올 수 있다.

```
from transformers import AutoConfig

model_name = 'bert-base-case'
model_config = AutoConfig.from_pretrained(model_name)
```

#### Config 사용시 주의사항
어떤 경우에는 config를 수정해서 사용하기도 하는데, configuration들 중에서도 바꿔도 되는 것이, 바꾸면 안되는 것이 존재한다.
- 바꾸면 안되는 config
    - pre-trained model을 사용할 때, hidden dim 등 이미 정해져있는 모델의 architecture setting은 수정하면 안된다.
    - 수정할 경ㅇ, 에러 발생 혹은 잘못된 학습으로 이어질 수 있다.
- 바꿔도 되는 config 
    - vocab의 경우, special token을 추가하면 추가한 token의 개수만틈 vocab을 늘려주어 학습해야 한다.
    - downstream task를 위해 몇 가지 config를 추가할 수 있다.
- 아래와 같이 vocab size를 늘려줄 수 있다.

```
model_name = 'bert-base-case'

model_config = AutoConfig.from_pretrained(model_name)
model_config.vocab_size = model_config.vocab_size + 2 # 또는 아래와 같이 사용할 수도 있다.
model_config = AutoConfig.from_pretrained(model_name, vocab_size = 28998)
```

### 1-3. Pre-trained Model
transformer를 활용하여 pre-trained model을 쉽게 불러올 수 있고, 해당 모델을 그대로 사용할 수도 있고, 추가적으로 학습을 진행하여 내 데이터에 맞게 사용할 수도 있다.
- `.from_config()` : config 그대로 모델을 가져오는 방법으로서 기학습 weight를 가져오지는 않는다.
- `.from_pretrained()` : model config에 해당하는 모델을 가져오는 방법으로서 기학습 weight를 함게 불러온다.

transformer는 두 가지 타입의 모델을 제공한다.
- 기본 모델
    - hidden state가 출력되는 기본 모델
- downstream task 모델
    - 일반적인 task를 쉽게 수행할 수 있도록 미리 기본 모델에 head가 부착된 형태로 설정된 모델이다.(e.g. AutoModelForQuestionAnswering, AutoModelForClassification)
    - output은 task에 적합한 dimension으로 미리 정의되어 있다. 즉, 우리의 task에 맞게 output dimension만 수정하여 활용하면 된다.

pretrained model의 경우 아래의 코드와 같이 불러와서 활용할 수 있다.

```
from transformers import AutoConfig, AutoModelforQuestionAnswering

model_name = 'bert-base-case'
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    config = model_config
)
```

### 1-4. Trainer
Trainer의 경우 아래의 구조로 간편하게 구성되어있고 활용할 수 있다.
- TrainingArguments 설정
- Trainer 호출
- 학습 및 추론


- 아래의 코드와 같이 trainer를 활용할 수 있다.

```
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = 'bert-base-case'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

batch_size = 32
args = TrainingArguments(
    model_name = model_name,
    evaluation_strategy = 'epoch',
    learning_rate = 5e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = 5,
    weight_decay = 0.01,
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer
)

trainer.train()
```

#### Trainer가 항상 좋을까? (중요)
- 버전이 바뀔 때마다 변동되는 사항이 많고 코드를 지속적으로 수정해야 한다.
- pytorch lightning이 대표적으로 위와 같은 legacy가 존재하고, huggingface의 transformers역시 예외는 아니다.
- **동작 원리를 살펴보는 과정이 매우 중요하다.**
- **Trainer 구조를 살펴보고, 내가 학습할 모델을 위한 Trainer를 만들어보기**
    - Trainer에 원하는 함수 오버라이딩 하여 수정하기 (general task에 적합)
    - Custom Trainer 만들어보기 (general task가 아닐 경우 유용)

### 1-5. Token 추가하기
간혹 모델의 성능을 높이기 위해 Special token을 추가하거나, domain에 특화된 단어를 추가해주는 방법이 있다.
- Special token을 추가하는 경우 해당 token이 special token임을 tokenizer에게 알려주어야 한다. 이 경우 `add_special_tokens()`를 활용할 수 있다..
- 일반 token을 추가하는 경우 역시 있을 수 있는데, 이 경우에는 `add_tokens()`를 활용할 수 있다.
- **tokenizer에 vocab을 추가했다면, pre-trained model의 token embedding size역시 vocab size에 맞춰 변경해주어야 한다.**
    - `model.resize_token_embedding`을 이용할 수 있다.
    - tokenizer의 `len()`을 활용하여 vocab size를 확인할 수 있고 이 출력값을 바탕으로 모델 사이즈도 늘려줄 수 있다.
- 코드는 다음과 같다.

```
model_name = 'bert-base-case'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    config = config
)

# special token 추가
special_tokens_dict = {'additional_special_tokens' : ['[S1]', '[S2]', '[S3]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# 일반 token 추가
new_tokens = ['COVID']
num_added_toks = tokenizer.add_tokens(new_tokens)

# config 수정
config.vocab_size = len(tokenizer)

# model token embedding size 수정
model.resize_token_embeddings(len(tokenizer)
```

### 1-6. [CLS] output 추출하기
- 여러가지 task를 수행할 때, 흔히 [CLS] token의 output을 바탕으로 classification, question answering 등이 수행된다.
- [CLS] embedding을 indexing을 통해 가져올 수 있지만, `.pooler_output`을 활용하면 보다 쉽게 가져올 수 있다.
- 다음의 코드와 같다.

```
model_name = 'bert-base-case'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
inputs = tokenizer('What can I do for you?!', return_tensors = 'pt')

outputs = model(**inputs)
cls_output = outputs.pooler_output # 이 embedding을 활용하여 다양한 task head에서 활용할 수 있다.
```

## 2. CLS 토큰은 문장을 대표할까?
- BERT 논문에는 [CLS] token이 문장을 대표하는 값으로 널리 알려져 있다.
- 하지만 의심의 여지 없이 [CLS] token이 문장을 대표하는걸까?
    > BERT의 저자 역시 [CLS]가 Sentence Representation을 보장하지는 않는다고 밝혔다.

- 이로 인해서, 다음과 같은 주의사항을 항상 확인해야한다.
    - **우리가 특정 task를 수행할 때, [CLS] 토큰이 당연코 문장을 대표할 것이라는 가정은 위험하다.**
    - **실험을 통해 우리가 수행하는 task에서 어떤 값에 대한 embedding이 중요한지를 확인해보도록 하자.**
    - [SBERT](https://arxiv.org/pdf/1908.10084.pdf)에서 이에 대한 설명이 이어진다.
        -  흔히, **avg나 CLS token embedding을 사용하는 것이 일반적이지만, Glove Embedding의 avg보다 성능이 낮았다고 한다.**
        -  즉, [CLS] token embedding은 sentence를 대표하지는 않는다.
        -  **input에 대한 representation을 추출한 후 pooling layer를 쌓아 maxpooling이나 average pooling을 수행하기도 한다.**
