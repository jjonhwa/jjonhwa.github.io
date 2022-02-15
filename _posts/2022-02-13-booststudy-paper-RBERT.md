---
layout: post
title: "[논문리뷰] RBERT: Enriching Pre-Training Language Model with Entity Information for Relation Classification"
categories: booststudy
tags: paper
comments: true
---
RBERT: [Enriching Pre-Training Language Model with Entity Information for Relation Classification](https://arxiv.org/pdf/1905.08284.pdf)를 읽고 이에 대하여 논의한다.

## 목차
- [1. Abstract](#1-abstract)
- [2. Introduction](#2-introduction)
- [3. Methodology](#3-methodology)
- [4. Experiments](#4-experiments)
- [5. Conclusion](#5-conclusion)

## 1. Abstract
- **Pre-Trained BERT Language Model**을 활용하고 **target entity를 통합**하는 모델을 제안한다.
- Target Entity를 찾고 그에 대한 정보를 Pre-Trained Model에 전달하고 **두 entity에 해당하는 encoding을 통합**한다.
- SOTA 모델의 성능을 개선한다.

## 2. Introduction
- Relation Classification task란 **명사 pair간의 의미 관계를 예**측하는 것이다.
- **Text Sequence 's'와 명사 Pair 'e1', 'e2'가 주어질 때, Objective는 'e1'과 'e2' 사이의 관계를 식별**한다.
- 일반적으로 다양한 NLP Application의 중간 단계로 사용되곤 하는 중요한 NLP task이다.
- 예를 들어, "kitchen"과 "house"사이의 Component-While 관계를 나타낸다.
    - **"The [kitchen] is the last renovated part of the [house]"**
- Pre-Training Language Model은 많은 NLP task에서 효과적인 개선을 보여주었고 특히, Pre-Trained BERT Model은 Multiple NLP task에 적용되어 11개의 task에서 SOTA를 달성하였다.
- **Pre-Training Language Model이 본 논문 작성당시에 "Relation Classification" 문제에는 적용되지 않았기에 본 작업과정에서 적용**한다.
- 두 개의 target entity의 위치를 식별하고 그 정보를 BERT Model에 전달하기 위하여 **text를 feeding하기 전에 target entity 전 후로 special token을 삽입**한다.
- 이후, **BERT Model의 output embedding에서 두 target entity들에 대한 encoding을 Classification을 위한 Multi-layer Neural Network의 입력으로 집어넣는다.**
- Relation Classification을 위해 PreTrained Model에 entity-level information을 통합하는 혁신적인 접근법을 제시한다.
- Relation Classification task에서 SOTA를 달성한다.

## 3. Methodology
### 3-1. PreTrained Model BERT
- PreTrained BERT Model은 Multi-layer bidirectional Transformer Encoder이다.
- BERT의 input representation의 디자인은 하나의 token sequence로 text sentence pair와 single text sentence 둘다를 표현할 수 있다.
- **각 token의 input representation은 해당 token, segment, position embedding의 summation으로 구성**된다.
- **"[CLS]" token은 sequence의 맨 처음에 붙는 token으로서 각 문장의 처음에 추가**된다.
- **두 문장이 있는 task에서는 "[SEP]" token이 두 문장을 구분하기 위한 special token으로 사용**된다.
- BERT의 경우, PreTraining Objective를 사용하여 사전학습된 Model Parameter를 사용한다.
    - Masked Langauge Model (MLM) (: input token 중 몇몇을 mask 처리한 후 이를 예측하도록 하는 task)를 활용하여 사전학습한다.
    - MLM의 경우 Deep Bidirectional Transformer를 적용하여 양방향의 정볼르 모두 활용할 수 있도록 한다.
    - Next Sentence Prediction (NSP) task를 적용하여 학습함으로서 Text-Pair Representation에 대하여 사전학습 시킨다. (본 과정은 이후 성능 개선에는 큰 효과가 없음이 실험적으로 증명되었다,)

### 3-2. Model Architecture
![4](https://user-images.githubusercontent.com/53552847/153740865-a2e13b17-bfaf-4d0e-8f7c-5933ceb27479.PNG)

- 문장 's'와 두 target entity 'e1', 'e2'가 주어졌을 때, **두 target entity에 대한 위치 정보를 포착하기 위하여 first entity에는 '$'를, second entity에는 '#'을 entity의 앞뒤에 삽입한다.**
- **[CLS] token을 각 문장의 맨 앞에 삽입한다.**
- 예를 들어, 앞에서 예시를 든 문장을 살펴보자면 다음과 같다.
    - "[CLS] THe $ kitchen $ is the last renovated part of the # house #"
-  각 Token에 대한 hidden state output을 H로 나타내며
    -  $$H_0$$ 는 [CLS]에 대한 hidden state output
    -  $$H_i$$ ~ $$H_j$$ 는 first entity token에 대한 hidden state output
    -  $$H_k$$ ~ $$H_m$$ 은 second entity token에 대한 hidden state output을 나타낸다.
- **first entity token과 second entity token에 대한 hidden state output은 i~j, k~m을 평균내어 계산**한다.
- 이후 3개의 final hidden state output (h0, h1(hi ~ hj에 대한 평균), h2(hk~hm에 대한 평균)은 **각각 tanh activation function과 fully connected layer를 통과**하게 된다.
- 이 때, **h1과 h2에 대해서는 같은 가중치 공유하여 사용하는 fully connected layer를 통과**하게 된다.
![1](https://user-images.githubusercontent.com/53552847/153740860-9a8d096d-a088-4a1c-93ac-da9e4ebb8f89.PNG)
![2](https://user-images.githubusercontent.com/53552847/153740862-24bd7089-137c-48e5-9f77-b2299e292fa0.PNG)

- fully connected layer를 통과하여 출력된 hidden state output을 **각각 $$H'_0$$, $$H'_1$$, $$H'_2$$라고 명명하며 이들을 concatenate하여 다시 한 번 fully connected layer를 통과하고 softmax layer를 통과**하게 된다.
![3](https://user-images.githubusercontent.com/53552847/153740864-860a20b0-a944-402b-94de-220678f3c52d.PNG)


- Loss Function으로서 기본적인 Cross Entropy를 활용한다.
- 매 fully connected layer 전에 dropout을 활용한다.

## 4. Experiments
### 4-1. Dataset and Evaluation Metric
- SemEval-2010 Task 8 Dataset
    - 9개의 Semantic Relation types
    - 1개의 Artificial Relation type (9개의 관계에 속하지 않는 관계)
- 9개의 Relation
    - Cause-Effect, Component-While, Content-Container, Entity-Destination, Entity-Origin, Instrument-Agency, Member-Collection, Message-Topic, Product-Producer
- 107171개의 Sentence를 가지고 있고 각각은 명사 e1과 e2를 포함한다.
- 관계에는 방향성이 있다. 예를 들어, Component-Whole(e1, e2)와 Component-Whole(e2, e1)은 서로 다르다.
- Dataset의 경우 8000개의 Training Dataset과 2717개의 Test Dataset으로 나뉘어져 있다.
- 1개의 추가적인 Label(Artificial Relation Type)을 제외하고 Macro-Averaged F1-Score를 바탕으로 계산하며 방향을 고려한다.

### 4-2. Parameter Setting
- Batch size: 16
- Max Sentence Length: 128
- Adam Learning Rate: 2e-5
- Number of Epochs: 5
- Dropout Rate: 0.1
- 매 Layer 전에 dropout을 추가한다.
- BERT의 경우, uncased-basic model을 활용한다.

### 4-3. Comparison with other Methods
- RBERT(본 논문의 모델)을 SVM, RNN, MVRNN, CNN+Softmax, FCM, CR-CNN, Attention-CNN, Entity Attention Bi-LSTM과 비교한다.
- SVM 방법의 경우 semeval-2010 task 8 competition에서 가장 좋은 결과를 가지는 전통적인 방법으로 rich feature를 사용한다.
- RBER의 경우 Macro F1이 89.25이고 이전의 solution들에 비해서 훨씬 좋은 성능임을 볼  있다.
![5](https://user-images.githubusercontent.com/53552847/153740867-798c37b3-e745-40cb-bd3a-724aa867d72c.PNG)


### 4-4. Ablation Studies
- PreTrained BERT Model 이외의 구성요소들의 구체적인 기여도에 대하여 이해하고자 추가적인 연구를 진행한다.

#### BERT-NO-SEP-NO-ENT
- 문장에서 두 entity의 앞 뒤에 있는 special token($, #)을 버린다.
- BERT의 출력 결과물에서 entity token에 대한 hidden vector를 사용하지 않는다.
- 즉, [CLS]에 대한 output vector만을 활용하여 classification을 수행한다.

#### BERT-NO-SEP
- 두 entity에 대한 Speical Token을 버린다.
- 두 entity에 대한 hidden vector output을 기존의 방식 그대로 합쳐서 classification을 수행한다.

#### BERT-NO-ENT
- 두 entity에 대한 hidden vector ouput을 버린다.
- 두 entity에 대한 Special Token은 그대로 유지한다.

#### 비교 결과
- 세 방법 모두 RBERT에 비해 성능이 안 좋다.
- BERT-NO-SEP-NO-ENT의 성능이 가장 좋지 않다.
- 본 비교를 통하여, special token의 사용과 entity에 대한 hidden vector를 사용하는 것 둘 다 본 논문의 접근 방법에서 중요하다는 것을 입증한다.
- special separate token이 없는 BERT의 경우 target entity들을 정확히 찾아낼 수 없게되고 이로 인해 핵심 정보를 잃어버릴 수도 있다.
- 즉, **speical separate token의 경우, target entity의 위치 정보를 BERT Model에게 전송하여 두 entity의 위치를 식별할 수 있도록 하기 때문에 성능의 개선에 도움**을 준다.
- 더불어, **target entity에 대한 hidden vector output을 [CLS] hidden vector output과 통합함으로서 정보를 풍부하게 해주고 정확도를 더 높게 만들어준다.**
![6](https://user-images.githubusercontent.com/53552847/153740868-d8124bcb-ca41-4047-8a30-dc9bc365bb6e.PNG)

## 5. Conclusion
- 본 논문에서, entity 정보와 함께 Pre-Trained BERT Model을 강화한 Relation Classification에 대한 접근을 개발하였다.
- target entity pair에 대한 special separate token을 더하고 classification을 위한 target entity representation을 활용한 sentence vector를 사용한다.
- SemEval-2010 benchmark dataset에 대하여 실험을 수행하고 그 결과 SOTA를 훨씬 능가한다.
- 추후 연구 과제로서 모델을 Distant Supervision에 적용하도록 확장하는 것이 있다.
