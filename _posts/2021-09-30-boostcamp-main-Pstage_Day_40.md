---
layout: post
title: "Second P-stage 4(Day 40)"
categories: boostcamp
tags: main
comments: true
---
NLP Relation Extraction task를 수행하는데 있어 Tips를 제공한다.

**부스트 캠프 39일차 학습 요약**
- **행사** : 오피스 아워 
- **P-Stage** : 베이스라인 모델 학습, KFold 구축, max_length 실험
- **피어 세션**

## 목차
- [1. Relation Extraction이란?](#1-relation-extraction)
- [2. Model](#2-model)
- [3. Special Token](#3-special-token)
- [4. Additional Layer](#4-additional-layer)
- [5. Pre-training](#5-pre-training)
- [6. Data Augmentation](#6-data-augmentation)
- [7. Input sentence](#7-input-sentence)

## 1. Relation Extraction이란?
- 하나의 문장이 있을 때, 특정 단어(entity)들 사이의 속성과 관계를 추론하는 문제
- 개체명 인식을 통해 entity에 대한 정보를 찾고, 관계 추출을 통해 그 두 개의 entity 사이의 관계를 출력해낸다.
- 왜 Relation Extraction을 할까?
    - 대규모 비정형 텍스트로부터 자동화된 정보를 수집할 수 있다.
    - 언어로부터의 핵심 정보 추출 및 요약이 가능하다.
    - 활용 범위
        - QA 시스템
        - 지식 그래프 구축

## 2. Model
- 다양한 Pre-trained Language Model을 사용할 수 있다.
- 한국어 모델은 다음과 같이 존재하며 각 task 및 data에 맞게 비교하며 사용할 수 있다.(2021 기준)
    - KLUE-RoBERTa_{small, base, large}
    - mBERT
    - KoBERT
    - KoGPT2
    - KoBart
    - KoElectra

## 3. Special Token
- Special Token을 활용하여 모델의 성능을 올릴 수 있다.
- "An Improved Baseline for Sentence-level Relation Extracion"(2021)으로부터 내용을 확인할 수 있다.
    - Entity Marker : Entity들의 위치 정보를 이용하여, [ENT], [/ENT]와 같은 entity special token을 각 단어 앞뒤에 붙여서 사용해볼 수 있다. (e.g. [ENT]이순신[/ENT])
    - Typed entity marker : Entity의 타입에 따른 서로 다른 special token을 사용할 수 있다. (e.g. [PERSON]이순신[/PERSON])
    - Typed entity marker (punct) : special token을 사용하지 않고 문장부호를 대신하여 사용할 수 있다. (e.g. @이순신@)

## 4. Additional Layer
- Additional Embedding Layer
    - Entity 유무에 따른 임베딩 레이어 추가
    - input_ids, token_type_ids, positional encoding과 더불어, entity_ids를 추가해서 input에 넣어준다.
- Additional Output Layer
    - Language Model을 중심으로 Layer를 더 쌓는다.
    - LSTM, GRU 등 RNN 모델을 마치막 classificaiton 단계에 추가해준다.
    - 모델을 더 깊게 쌓을 수 있고, 모델을 통해 나온 Embedding을 좀 더 sequence 사이의 관계를 모델링 할 수 있다.

## 5. Pre-training
- 사전 학습된 언어 모델을 주어진 대회 데이터 셋 내의 문장들로 한 번 더 사전학습 한 다음, fine-tuning을 진행한다.
- 언어 모델이 RE 데이터 셋 내의 문장들에 대하여 적응할 수 있또록 도와준다.

## 6. Data Augmentation
- 부족한 데이터 문제를 해결하기 위해 데이터를 증강 시키는 것
- Text는 해결하고자 하는 Task에 따라 필요한 데이터들이 모두 다르기 때문에, 하나의 NLP Task를 위해 충분한 데이터를 확보하는 데 많은 시간과 비용이 소요된다.
- 최근에는 다양한 Text Augmentation 기법들이 논문에서 소개되어 적은 데이터에서도 좋은 성능을 낼 수 있도록 연구가 활발히 진행되고 있다.
- But, Text Augmentation은 생각보다 어려운 작업이다.
- 원래의 문장이 가지고 있는 의미를 보존한 채로, 외적인 구조만 변경해야하기 때문이다.
- 또한, 주어진 문장의 의미를 완벽하게 정의내리기도 어렵고, 기존&생성된 2개의 문장의 객관적인 슈다로를 판단하여, Augmentation이 올바르게 됬는지 판단하기도 어렵다.

### 6-1. EDA
- Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
- [EDA 논문](https://arxiv.org/abs/1901.11196)에서 논문을 확인할 수 있다/
- Text Classification 분야에서 간단한 전처리 기법을 통해 Augmentation을 진행할 수 있다.
- 전체 Dataset의 50%만을 가지고도 전체 Dataset을 사용했을 때와 같은 성능을 보였다고 논문에 기재되어 있다.
- 외부 데이터 혹은 이전 학습된 모델을 사용하지 않는다.

#### EDA 기법
- SR(Synonym Replacement) : 문장에서 불용어를 제외한 임의의 단어를 선택한 후, 동의어로 대체
- RI(Random Insertion) : 문장의 임의의 위치에 임의의 단어를 삽입
- RS(Random Swap) : 문장의 임의의 두 단어의 위치를 스왑
- RD(Random Deletion) : 문장 내의 임의의 단어를 삭제

#### EDA 구현
- 카카오 브레인에서 공개한 pororo 라이브러를 사용하여 구현할 수 있다.
- [카카오 브레인 code](https://github.com/kakaobrain/pororo/blob/master/examples/)에서 예제를 진행할 수 있다.

#### EDA 방법론
- SR (Synonym Replacement)
    - 문장이 들어온다.
    - 문장에 대한 단어들의 품사를 하나하나씩 살펴본다.
    - 명사인 단어일 경우, 이를 Word2Vec 공간 상에서 가장 비슷한 단어를 찾아 그 단어로 교체해준다.
- RI (Random Insertion)
    - 새롭게 추가할 단어를 입력해준다.
    - 주어진 문장을 모두 단어로 쪼갠다.
    - 새로운 단어를 추가할 문장 위치를 랜덤으로 선택하여 삽입한다.
- RS (Random Swap)
    - 문장 내에서 임의의 두 단어를 고른다.
    - 두 단어의 위치를 바꿔준다.
- RD (Random Deletion)
    - 문장 내에서 임의의 단어 하나를 선택한다.
    - 이 단어를 문장에서 삭제한다.

#### 실험 결과
- 데이터 셋이 작을 때, 높은 성능 향상을 보인다.
- 데이터 셋이 충분히 많을 경우, 아주 미미한 향상이 나타난다.

### 6-2. AEDA
- An Easier Data Augmentatio Technique for Text Classification, 2021에서 소개된 Data Augmentation 기법이다.
- 원래 데이터에 무작위로 문장 부호 (.,?;:!)를 넣고, 이를 학습 데이터로 학습한다.
- 추가하는 문장 부호의 개수는 [0, (전체 단어 개수의 1/3)] 범위 내에서 무작위로 추출한다.
- 실험 결과, EDA보다 전반적으로 높은 성능을 보인다고 한다.

### 6-3. Round-trip translation
- Back translation이라고도 한다.
- Understanding Back-Translation as Scale을 통해 소개된 Data Augmentation 기법이다.
![29](https://user-images.githubusercontent.com/53552847/135759034-57e6f88b-eaf8-41b8-bffc-b48608d3d8c2.PNG)
- 기계 번역 시스템을 이용하는 Data Augmentation 기법이다.
- 원래의 학습 샘플을 다른 언어로 번역한 뒤, 다시 원래의 언어로 번역해서 새로운 학습 샘플을 생성한다.
- 원래 문장과 동일한 의미를 가지지만 표면적으로는 다른 문장을 만드는 과정으로 이해할 수 있다.
![30](https://user-images.githubusercontent.com/53552847/135759033-131de1d5-a7b6-4dc0-b84a-ad45ee7619cc.PNG)
- Pororo library를 통해, 위의 코드처럼 진행할 수 있다.
- 구글 번역기, 파파고 등 다양한 플랫폼에서 제공하는 번역 API를 사용할 수도 있고, 직접 번역 모델을 학습한 뒤 inference를 진행할 수도 있다.
- 실제로 back-translation을 할 때는, 중간에 어떤 언어로 변환하는지에 따라 성능이 다르게 나타난다.
- 번역기의 성능에 의존적이다.
- 번역을 하는 과정에서 entity가 날아갈 수도 있기 때문에, 번역된 문장을 어떤 기준으로 사용할 것인지를 정하는 것도 중요하며, 혹은 entity가 날아가지 않게 전처리하는 방법을 추가하는 것 역시 중요하다.

## 7. Input sentence
- input sentence를 넣어주는 방식에 따라 여라가지 종류로 나뉠 수 있다.
- RE(relation extraction)의 경우는 Again Task, Multi, Single 등의 방법을 사용할 수 있는데 이 3가지 방법에 대하여 가볍게 언급하도록 한다.
    - Again Task : (원문, entity1, entity2)가 있을 경우 이들을 "[CLS] entity1 [SEP] entity2 [SEP] 원문 [SEP]"의 방식으로 묶어서 input으로 넣어준다.
    - Multi : (원문, entity1-entity2 사이의 관계를 묻는 질문 문장) 을 "[CLS] 원문 [SEP] 질문 [SEP]" 형식으로 만들어준다.
    - Single : (원문, 질문)을 "[CLS] 원문. 질문 [SEP]"의 형식으로 만들어 input으로 넣어준다.






