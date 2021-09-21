---
layout: post
title: "NLP Ustage Day 10 (Day 34)"
categories: boostcamp
tags: main
comments: true
---
Advanced Self-supervised Pre-training Model에 대하여 학습한다.

**부스트 캠프 34일차 학습 요약**
- **학습** : Advanced Self-supervised pre-training models
- **행사** : 마스터 클래스(주재걸 교수님)
- **피어 세션**

## 목차
- [1. GPT-2](#1-gpt-2)
- [2. GPT-3](#2-gpt-3)
- [3. ALBERT](#3-albert)
- [4. ELECTRA](#4-electra)
- [5. Light weight Models](#5-light-weight-models)
- [6. 34일차 후기](#6-34일차-후기)
- [7. 해야할 일](#7-해야할-일)

## 1. GPT-2
- Model Architecture는 GPT-1과 크게 다르지 않다. GPT-1에 대한 설명은 [GPT-1](https://jjonhwa.github.io/boostcamp/2021/09/16/boostcamp-main-Ustage_Day_33/#2-gpt-1)에서 확인할 수 있다.
- GPT-1에서 Transformer Model에서의 self-attention block을 점점 더 많이 쌓아서 모델을 키웠고, pre-training task는 여전히 LM을 사용하여 다음 단어 예측 task로 사용하였다.
- Training data의 경우 훨씬 증가된 사이즈의 40GB의 텍스트를 사용하였고, 특히, 질 높은, 잘 쓰여진 글을 Dataset으로 사용하여 효과적으로 다양한 지식을 배울 수 있도록 유도하였다.
- 여러 downstream task에 대하여, 생성 모델이라는 Language generation task에서 zero-shot setting으로서도 잘 다뤄질 수 있다는 잠재 능력도 보여주었다.
- 즉, Pre-training된 모델을 활용하여, 주어진 첫 문단이 있을 경우 이 문단을 이어받아 다음 단어 그리고 그 다음 단어를 쭉쭉쭉 예측해서 어떤 하나의 긴 글을 완성할 수 있는 능력을 가지게 된다.

### 1-1. Motivation
- GPT-2의 Motivation은 decaNLP이다. 
- The Natural Language Decathlon: Multitask Learning as Question Answering이라는 논문으로부터 동기를 얻었다.
- 이 논문의 핵심은, 서로 다른 downstream task의 경우 서로 모델 구조가 상이하고, 즉, output의 형태가 모두 다르지만, 모든 종류의 자연어 task들은 질의응답 형태로 바뀔 수 있고, 이를 통합된 자연어 생성의 형태로 다양한 task들을 통합해서 학습을 했다는 것이다.
- 예를 들어, 긍,부정을 예측하는 task로서 I love this movie를 분류하고자 할 때, 이 이후에 질문을 하나 추가하여, What do you think about this movie?라는 질문에 대한 대답을 찾는 방식으로 바꿀 수 있다.
- Summarization의 경우, 문단의 마지막에 What is the summarization of paragraph?라는 질문을 두어 질문에 대한 대답을 찾는 방식으로 바꿀 수 있다.

### 1-2. Dataset
- 굉장히 많은 Dataset과 그 중 굉장히 질 높은 글을 선별하여 잘 쓰여진 글로부터 생성된 Dataset을 사용하여 지식을 효과적으로 배울 수 있도록 설계하였다.
- Reddit(community website)은 질문을 올리고 이 질문에 대한 reply를 달게 되는 방식으로 구성되어 있는데, reply에서 외부링크가 있으며, 좋아요가 3개 이상 달린 reply의 경우 잘 쓰여진 글일 가능성이 높다고 판단하여 이러한 reply의 링크의 document를 수집하여 이를 바탕으로 Dataset을 구성하였다.
- Byte Pair Encoding(BPE)라는 subword level에서의 word embedding을 도출하고, 해당 사전을 구축해줄 수 있는 알고리즘을 사용했다.

### 1-3. Model
- Layer Normalization이 특정 위치에서 하나 더 위로 가거나 아래로 가거나하는 세부적인 차이점이 있다. 즉, GPT-1과 비교해서 Layer Normalization의 위치가 변화했다는 차이점이 있다.
- 각 Layer들은 random initialization을 할 때, layer가 위로 가면 갈수록 random initialization되는 값을 더 작게 만들어서, layer가 위로 가면 갈 수록 사용되는 선형변환에 해당하는 값들이 점점 0에 가까지워지도록 만들어서 위쪽의 layer가 하는 역할을 점점 더 줄어들 수 있도록 모델을 만들었다.

### 1-4. Question Answering
- decaNLP로부터 얻은 모든 task들은 QA형식으로 바뀔 수 있다는 사실에 입각하여 QA task에 대한 실험을 진행하였다.
- 실제로 해당 task에 대한 실험을 진행하기 위해서는 labeled data를 사용하여 fine-tuning하지만, 본 연구에서는 zero-shot learning의 setting에서 이 task에 대한 예측을 수행함에 있어 학습데이터를 하나도 쓰지 않고 어느 정도의 예측 성능이 나오는지를 test했다.
- 이를 통해, 대략 55점의 f1-score가 도출됬고, 이는 해당하는 downstream task를 위한 data를 가지고 학습한 후 fine-tuning을 하여 달성한 성능에는 못 미치지만 어느 정도 예측 가능하다는 가능성을 보였다는 사례가 되었다.

### 1-5. Summarization
- 기존의 Summarization의 경우 article을 먼저 주고 마지막에 (TL;DR)이라는 단어를 주어 학습데이터의 많은 글들 중에 TL;DR이 나오면 앞쪽에 있던 글을 한 줄 요약하는 형태로 진행된다.
- summarization의 경우에서도 마찬가지로, labeled data를 활용해 GPT-2를 fine-tuning하지 않고도 zero-shot setting으로 요약을 수행할 수 있게 된다.

### 1-6. Translation
- translation은 주어진 문장이 있을 때, 그 마지막에 번역하고 싶은 어떤 문장 (예들 들어, in French, thesy say in Frecn 등)들을 붙여주면 앞서 나온 문장들을 해당하는 언어로 잘 번역하는 사례들을 보여준다.

## 2. GPT-3
- GPT-2를 훨씬 더 개선한 모델이다.
- Model Architecture 측면에서는 특별한 점은 없으며, GPT-2 모델 사이즈 혹은 Parameter의 수를 비교할 수 없을 정도로 많이 가지게 하도록 self-attention block을 훨씬 더 많이 쌓은 형태이다.
- 더 많은 data와 더 큰 batch size(3.2M)을 통해 학습을 진행했더니 서능이 점점 더 좋아졌다.
- GPT-2에서 보여주었던 zero-shot setting에서의 가능성을 놀라운 수준으로 끌어올렸다.
![35-1](https://user-images.githubusercontent.com/53552847/133793928-5a4fdfd4-ad8e-45f1-ba28-bde0d27983d6.PNG)
- zero-shot setting에서 위의 이미지에서와 같이 task description을 주고 번역할 단어를 주면 번역을 해낸다. 즉, 번역 task에 대한 학습 데이터를 전혀 사용하지 않고 번역을 해내는 샘이다.
- 여기서 흥미로운 점은, 위의 이미지에서 처럼, task description을 주고 예시를 하나 주게 되면, 이를 one-shot setting이라고 하는데 이는 결국 학습데이터로서 하나의 데이터만을 주는 의미이고, 이렇게 해서 번역을 수행했을 때, 번역을 더 잘 수행했다는 사실을 발견했다.
- 더불어, one shot이 아닌 few shot setting으로 진행했을 경우 더 유의미한 높은 성능을 낼 수 있음을 알 수 있었다.
- 이러한 zero-shot, one-shot, few-shot setting으로 부터의 더욱 의미있는 성과는, 기존에 downstream task를 수행하기 위해서는 모델에서 마지막 output layer를 task에 맞게 수정하고 이를 downstream task를 위한 data를 활용해 학습을 하여 inference를 수행하였지만 GPT-3에서는 모델 자체를 전혀 변형하지 않은 채 단지 inference 과정 중에 task description과 examples를 추가하여 inference를 수행해 유의미한 성능을 도출했다는 것이다.
- 이는 별도의 fine-tuning 과정없이 GPT-3 모델을 그대로 가져와서 예시를 보여주고, 하나의 sequence 내에서 하나의 패턴을 동적으로 빠르게 학습한 후 task를 수행하는 GPT-3의 놀라운 능력을 보여준다.
![36](https://user-images.githubusercontent.com/53552847/133711830-3e6a2cf0-9f84-4ea7-b915-8ee70e8430bc.PNG)
- 위의 사진에서 처럼, 실제로 GPT-3에서 보여준 다양한 결과 중에서 모델 사이즈를 키우면 키울수록 성능이 올라가는 gap이 점점 더 벌어지는 사실을 보여주었다.
- 큰 모델을 사용할수록 모델의 동적인 적응능력이 훨씬 더 뛰어나다는 사실을 보인게 된다.

## 3. ALBERT
- A Lite BERT for self-supervised learning of Language representations
- 기본적으로 앞서 보여준 pre-training model들은 대규모의 메모리 요구량과 많은 학습에 필요로 하는 모델 Parameter를 가지는 형태로 점점 더 발전을 해왔지만, 이로 인해 더 많은 GPU 메모리를 필요로하고 더 많은 학습데이터와 학습 시간을 필요로 했다.
- ALBERT는 기존의 BERT가 가지던 비대한 size를 성능의 큰 하락없이 오히려 성능이 개선되는 형태를 유지하면서 모델의 size를 줄이고 학습시간도 빠르게 만들 수 있는 추가적으로 새로운, 변형된 형태의 문장 레벨의 self-supervised learning의 pre-training task를 제안하였다. 
![37](https://user-images.githubusercontent.com/53552847/133711833-7ecf24fc-431d-4565-8745-b4af65f046bc.PNG)

### 3-1. Factorized Embedding Parameterization
- 기존의 BERT, GPT 등의 모델에서는 Residual connection에 의해서 입력에 주어지는 word embedding dimesion수와 residual connection으로 인한 다음 layer의 self-attention에서도 동일한 dimension을 가지게 되고, 이러한 dimension이 작게되면 정보를 담을 수 있는 공간 자체가 작아지는 단점이 있을 수 있고 크게되면 model size가 너무 비대해져 이에 따른 연산량도 증가하게 될 수 있따.
- self-attention block을 쌓아가는 것은 좀 더 high level의 의미론적으로 유의미한 정보를 추출해 나가는 과정이라고 할 수 있는데, layer에서 각 word가 문장 내에서 가지는 관계라던지 여러 contextual한 정보를 고려하지 않고 각 word별로 독립적으로 상수형태의 vector로서 주어지는 embedding layer가 있을 때, 이 embedding layer에서의 word가 가지는 정보는 전체 문장을 고려해서 각 word별로 만든 encoding hidden state vector들에 비해서는 상대적으로 훨씬 더 적은 정보만을 가지고 있고, 그 정보만으로도 충분할 수 있다.
- 이러한 의미에서, ALBERT model은 embedding layer의 dimension을 줄이는 추가적인 기법을 제시했다.
![38](https://user-images.githubusercontent.com/53552847/133711839-92e3e5a3-bdf7-447c-8aad-6453f261ec4b.PNG)
- self-attention block에서 사용하는 고정된 dimension이 위의 그림처럼 4차원이라고 할 때, 일반적으로 word가 가지는 embedding vector에 positional embedding을 더해주는 형태로 진행된다.
- embedding layer에서의 dimension은 각 word들이 self-attention을 통해 출력되는 encoding vector의 dimension과 동일해야하는데, 이 때 self-attention의 입력으로 주는 word-embedding의 차원을 줄여서 pararmeter와 계산량을 줄일 수 있는 기법을 제시한다. 이 경우, 위의 그림의 오른쪽 부분과 같이 작동한다.
- word embedding vector가 기존에 4차원이었는데, 각 word별로 2차원만을 가지는 word embedding vector를 구성했다라고 하는 경우, 4차원 벡터를 입력으로 받는 self-attention block의 차원 수를 맞춰주기 위해, 추가적으로 하나의 fully connected layer를 더 두고, 그 layer를 통해 4차원으로 dimension을 늘려준다.
- row-rank matrix factorization이라는 기법을 통해, 4차원의 embedding vector를 2차원의 vector와 이에 적용되는 선형변환 matrix가 있어, 전체적인 parameter의 수를 줄여주는 방식으로 이해할 수 있다.
- 계산량은 실제로 100 dimension을 15 dimension + Linear Transformation으로 번형했다고 할 때, 기존의 "word의 개수x100"에서 "word의 개수x15 + 15x100"가 되어 parameter의 수가 현저히 줄어드는 것을 알 수 있다.

### 3-2. Cross layer Parameter Sharing
- 적층해서 쌓는 self-attention block에서, 서로 다른 self-attention block에서 존재하는 선형변환 matrix들을 공통적인 혹은 shared된 그런 parameter들로 구성하면 어떨지, 즉 $$W_q$$, $$W_k$$, $$W_v$$, $$W_o$$ matrix를 shared된 하나의 set의 Parameter로 동일하게 적용하는 layer들로 기존의 BERT model을 개선한 것이다.
![38-1](https://user-images.githubusercontent.com/53552847/133802549-e748f925-81ba-402f-a42e-83cb125cab84.PNG)
- output layer에 해당하는 feed forward network parameter만 share 했을 때, attention을 수행하기 위해 사용되는 $$W_q$$, $$W_k$$, $$W_v$$ parameter만을 share했을 때, 그리고 이 모두를 share했을 때로 나뉘어서 각각에 대해 모델을 구성하여 실험을 하였고, 이러한 결과 위의 표와 같은 결과가 도출되었다.
- 그 결과, share를 하지 않는 original BERT에 비해서, ALBERT는 parameter sharing을 통해 Parameter의 수는 훨씬 줄어들지만, 성능은 크게 하락하지 않게 하도록 한다.

### 3-3. Sentence Order Prediction
- BERT는 pre-training 과정에서 MLM과 Next Sentence Prediction을 통해서 진행되는데, BERT 이후의 후속 연구들에서 Next Sentence Prediction이라는 task는 BERT 모델에서 실용성이 없다라고 했고, 이 task를 아예 빼고 MLM만을 수행했을 때, 다양한 task들에 대하여 fine-tuning해서 나타나는 성능들이 next sentence prediction을 포함했을 때 나온 BERT 성능에 비해 그다지 떨어지지 않다는 사실을 지적했다.
- ALBERT에서 실제로 실효성이 많이 없는 Next Sentence Prediction task를 좀 더 유의미한 task로 바꾸어 실제 pre-training task 과정에서 그 모델이 좀 더 유의미한 정보 혹은 지식을 깨우칠 수 있도록 하는 task로 확장했다.
- 이 새로운 task는 next sentence인지 아닌지를 prediction하는 것이 아니라, 항상 같은 document 내에서 연속적으로 등장하는 두 문장을 가져오고 그 두 문장을 원래 순대로 혹은 역순으로 concat하여 이 연결된 두 문장의 sentence order가 맞는지 혹은 역순인지를 구별하는 binary classification task로서 next sentence prediction task를 sentence order prediction으로 변형하여 사용하였다.
(계속)
![39](https://user-images.githubusercontent.com/53552847/133711840-da1e3aba-95fb-4238-bda2-66d0146f03dd.PNG)

### 3-4. GLUE Results
![40](https://user-images.githubusercontent.com/53552847/133711841-78cf206a-2291-467a-9196-cb1dfe3c5e24.PNG)

## 4. ELECTRA
![41](https://user-images.githubusercontent.com/53552847/133711842-fffa39e2-faca-4fa6-9dd8-d4da574ad588.PNG)
![42](https://user-images.githubusercontent.com/53552847/133711843-037806b6-1617-453a-a91f-7f0637d0eae2.PNG)

## 5. Light-weights Models
- pre-trained된 모델을 다양한 방식으로 고도화하는 연구들이 활발하게 연구되고 있으며, 그 중 하나로 모델 경량화 연구가 있다.
- 기존의 BERT, GPT-2, GPT-3, ELECTRA 등의 모델들이 self-attention block을 점점 더 많이 쌓음으로서 더 좋은 성능을 냈지만, pre-trained하기 위해서 더 많은 GPU resource와 시간, 계산량이 필요했고, 이로 인해 실제 현업에서 사용하기에는 어려움이 있다.
- 경량화 model은 이렇게 비대해진 모델을 좀 더 적은 layer 수나 parameter 수를 가지는 경량화된 모델로 발전시키는 혹은 확장하는 형태의 연구이다.
- 경량화 모델의 연구추세는 기존의 큰 사이즈 모델이 가지던 성능을 최대한 유지하면서 모델의 크기와 모델의 계산 속도를 빠르게 하는 것에 초점이 맞추어져 있다.
- 경량화된 모델은 CLOUD 서버나 고성능의 GPU resource를 사용하지 않고서도 가령, 휴대폰 등의 소형 device에서도 모델을 load해서 더 적은 전력 소모 혹은 배터리 소모량으로 빠르게 계산을 수행하고자 할 때 사용된다.
- 모델을 경량화하는 방식은 다양하지만 여기서는 Distillation 기법을 활용한 두 개의 모델만 짧게 설명하고자 한다.

### 5-1. DistillBERT
- hugging face에서 발표한 논문이다.
- teacher model, studenet model이 있으며, teacher model은 student model을 가르치는 역할을 하며, student model은 teacher model에 비해 layer 수나 parameter 측면에서 더 작은 경량화된 형태의 모델이다.
- student model은 teacher model이 내는 여러 output이나 pattern을 잘 묘사할 수 있도록 학습이 진행된다.
- 작은 사이즈의 student model은 teacher model이 어떤 주어진 문장에서 주어진 문장에 대해서 MLM을 수행할 때, output vector를 softmax를 통과시키고 이를 vocabulary 상에서의 확률분포를 예측값으로 주게 되는데 여기서 나온 예측값을 ground truth로 사용하는 것이다. 즉, teacher model에서의 예측값을 student model의 ground truth로 사용하는 방식이다.
- 이렇게 하여 student model이 teacher model이 하는 여러 행동 혹은 예측 결과를 최대한 잘 묘사할 수 있도록 학습이 진행된다.

### 5-2. Tiny BERT
- Tiny BERT 역시 teacher model과 student model이 있지만, distillBERT와 달리, Tiny BERT에서는 실제 target distribution을 ground truth로서 student model을 적용하고 teacher model을 담도록 학습하는 방식을 채택한다.
- 이에 더하여, 어떤 Embedding layer와 각 self-attention block이 가지고 있는 $$W_q$$, $$W_k$$, $$W_v$$ 등의 attention matrix, 그 결과로 나오는 hidden state vector까지 유사해질 수 있도록 student network가 teacher network를 담을 수 있도록 하는 형태로 학습을 진행한다.
- 즉, teacher model에서의 최종적인 예측값 뿐만 아니라 그 과정의 중간 결과물들이 있는데, 여기서 student model의 중간 결과물이 teacher model의 중간 결과물에 최대한 가까워질 수 있도록 MSE loss를 사용하여 학습을 동시에 진행하게 된다.
- 하지만, 이렇게 할 경우, student model의 hidden state vector가 teacher model의 hidden state vector의 차원수보다 작을 수 있으므로, 차원으로 인한 loss를 적용하기가 어려울 수 있다.
- 이에 대하여, student model과 teacher model의 hidden state vector가 유사한 형태를 띌 수 있게 만들어 loss를 적용하기 위하여, teacher model에 hidden state vector가 최종적으로 fully connected layer를 한 번 더 거쳐서 student hidden state vector와 차원이 맞도록 하여 dimension으로 인한 miss-match를 해결하였다.
- Tiny BERT의 핵심은, student model에서의 최종 output으로 나타나는 예측값 뿐만 아니라 중간 결과물도 teacher model을 담을 수 있도록 학습한다는 것이다.

## 6. 34일차 후기
NLP level 2의 U-stage도 끝났다. 하지만... 공부를 하면 할 수록 해야할 공부가
마치 지수함처럼 불고 있는 것만 같다 ㅜㅜ.

해야할 일이 산더미이지만 차분히 하나하나씩 해결해 나가도록 하자! 출발선은 뒤쪽에 있을지 모르더라도 천천히 꾸준히해서 결승선에 도달하는 것만큼은 비슷할 수 있도록 노력해야겠다~!

## 7. 해야할 일
- GPT-2에서 layer의 index에 비례하여 random initialization 되는 값을 더 작게 만들면 위쪽의 layer로 갈수록 하는 역할을 점점 줄어들게 만든다고 하는데 왜 그러는걸까? 
- 'layer에서 각 word가 문장 내에서 가지는 관계라던지 여러 contextual한 정보를 고려하지 않고 각 word별로 독립적으로 상수형태의 vector로서 주어지는 embedding layer가 있을 때, 이 embedding layer에서의 word가 가지는 정보는 전체 문장을 고려해서 각 word별로 만든 encoding hidden state vector들에 비해서는 상대적으로 훨씬 더 적은 정보만을 가지고 있고, 그 정보만으로도 충분할 수 있다.'가 어떤 의미일까? contextual한 정보를 고려하지 않으면 그 정보만으로 충분할 수가 있나?
- row-rank matrix factorization의 원리는 무엇인가?(Factorized Embedding Parameterization)
- Cross layer Parameter Sharing에서 shared된 하나의 set의 Parameter란 무엇인가 그리고 Cross layer Parameter Sharing의 개념에 대해서 더 깊게 이해해보도록 하자.
