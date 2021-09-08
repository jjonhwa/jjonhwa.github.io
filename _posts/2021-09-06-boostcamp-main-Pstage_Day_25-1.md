---
layout: post
title: "NLP Ustage Day 1-1 (Day 25)"
categories: boostcamp
tags: main
comments: true
---
자연어 처리의 가장 간단한 모델에 대하여 학습한다.

**부스트 캠프 25일차 학습 요약**
- **행사** : MeetUP(w. 피어), MeetUP(부캠에서 살아남기)
- **학습** : Bag-of-Words, Word Embedding
- **피어 세션**

## 목차
- [1. Natural Language Process](#1-natual-language-process)
- [2. Bag-of-Words](#2-bag---of---words)

## 1. Bag-of-Words
### 1-1. Natural Language Processing
- NLP 분야의 최신기술 및 연구결과가 발표되는 것은 대표적으로 ACL, EMNLP, NAACL이 있다.

#### low-level parsing
- Tokenization : 주어진 말뭉치 Corpus에서 token이라고 불리는 단위로 나누는 작업
- stemming : 어간 추출

#### Word and phrase level
- Named entity recognition (NER)
    - 단일 단어 혹은 여러 단어로 이루어진, 고유 명사를 인식하는 task
    - NewYork Times라는 단어가 있을 때, 이를 3개의 단어가 아닌 하나의 고유명사로 인식하는 것이다.
- part-of-speech(POS) tagging
    - word들이 문장 내에서의 어떤 품사, 성분인지를 파악한다.
    - wrod가 주어진 문장에서 주어, 동사, 목적어, 부사, 형용사 등 어떤 역할을 하는지 인식하는 task
- 이밖의 noun-phrase chunking, dependency parsing, coreference resolution 등이 있다.

#### Sentence level
- Sentiment Analysis
    - 주어진 문장이 긍정 혹은 부정인지 예측하는 감정분석
    - 예를 들어, I love this movie -> 긍정 / I hate this movie -> 부정
- Machine Translation
    - 주어진 언어의 문장을 전체적으로 이해하고, 다른 언어로의 번역 및 각 언어의 문법을 고려한 어순으로 출력하는 task
    - 예를 들어, I study math -> 나는 수학을 공부한다.

#### Multi-sentence and paragraph level
- Entailment prediction
    - 두 문장간의 논리적인 내포 혹은 모순관계 예측
    - 예를 들어, "어제 존이 결혼을 했다."와 "어제 최소한 한 명은 결혼을 했다."라는 문장이 주어졌을 때, 1이 참인 경우에 2는 자동으로 참이된다.
    - 또한, "어제 존이 결혼을 헀다."와 "어제 아무도 결혼하지 않았다."가 주어졌을 때, 논리적인 모순 관계를 가진다.
    - 이처럼, 두 문장 사이의 논리적인 관계에 대해서 예측하는 task이다.
- Question Answering
    - 독해기반의 질의 응답
    - 구글에 질문을 입력했을 때, 질문에 대한 답에 해당하는 정보를 나타내 준다.
    - 질문에 대한 검색 결과를 뽑아낸 후 질문을 분석하여 이 질문에 대한 결과를 정확히 알아내 정답을 사용자에게 제시해준다.
- Dialog Systems
    - 챗봇과 같은 대화를 수행할 수 있는 기술 
- Summarization
    - 주어진 문서를 한 줄 요약의 형태로 나타낸다.

### 1-2. Text mining
- Text mining 분야의 최신기술 및 연구결과가 발표되는 것은 대표적으로 KDD, The WebConf, WSDM, CIKM, ICWSM이 있다.

#### Extract usful information and insights from text and document data
- Analyzing the trends of AI-related keywords from massive news data
- 기존의 빅데이터로부터, 트렌드를 분석
- 예를 들어, 회사에서 상품을 출시했을 때, 상품을 바탕으로 오고간 대화들을 분석함을서 상품을 사용해본 소비자들의 반응을 확인하는 task이다.

#### Document clustering
- Clustering news data and grouping into different subjects
- 서로 다르지만 비슷한 의미를 지닌 키워들을 Grouping 해서 분석할 필요가 생기는데 이를 자동으로 수행하기 위해 Copying Modeling 혹은 문서 군집집화등의 기술을 사용하여 진행한다.

#### Highly related to computational social science
- Analying the evolution of people's political tendency based on social media data

### 1-3. Information Retrieval
- Information Retrieval 분야의 최신기술 및 연구결과가 발표되는 것은 대표적으로 SIGIR, WSDM, CIKM, RecSys가 있다.

#### 검색기술을 연구하는 분야
- 검색 키워드에 대한 정보를 나타내 주는 기술
- 현재 검색기술은 어느 정도 성숙한 상태에 이르렀다고 볼 수 있다.

#### 추천 시스템
- Youtube, 음악 플랫폼에서 영상 혹은 음악을 추천해주는 기술
- 사용자가 수동으로 검색할 과정을 자동으로 사용자가 좋아할 법한 영상 혹은 음악을 선제해준다.
- 상업적으로 굉장히 큰 Impact를 가지고 있다.
- 개인화된 광고 혹은 상품 추천에 이르기까지 다양한 분야에서 활발히 사용되고 있다.

#### Highly related to computational social science
- The area is not actively studied now
- It has evolved into a recommendation system, which is still an active area of research

### 1-4. NLP의 최근 발전 과정 (Trends)
- Word Embedding
    - 단어를 벡터로 나타내는 것
    - 단어뿐만 아니라 순서를 인식할 수 있어야 한다.
- Sequence Data Processing
    - Sequence Data에 특화된 모델 구조로서 RNN이라는 자연어처리의 핵심 모델로 자리잡았다.
    - RNN 중 LSTM, GRU 등의 모델이 많이 사용되어 왔다.
    - 2017년도에 RNN 기반 자연어 처리 모델 구조를 Self-Attention이라는 모듈로 완전히 대체할 수 있는 Transformer라는 모델이 등장하였다.
    - 다양한 자연어 처리 분야에서 Transformer라는 모델은 큰 성능향상을 가져오게 되었다.
    - 현재 대부분의 자연어 처리를 위한 딥러닝 모델들은 Trnasformer 모델을 기본 구조로 하고 있다.
    - Transformer 모델은 기계번역 task를 위해 처음 제안되었는데, 이 뿐만 아니라 시계열 예측, 영상처리, 신약개발, 신물질개발 등의 다양한 분야에서 활발히 적용되고 있다.
- 자연어처리에서의 Self-Supervised Learning
    - 입력문장이 주어졌을 때, 입력 문장 중 일부 단어를 가리고 그 단어를 맞출 수 있도록 하는 task
    - 이를 활용한, 사전 학습 혹은 Pre-Trained Model의 예시로서 BERT, GPT 모델들이 존재한다.
    - 높은 GPU resource가 필요하므로 막강한 자본력과 데이터가 뒷받침되어야 진행할 수 있다.

## 2. Bag-of-Words
- 자연어 처리 및 Text Mining 분야에서 딥러닝 기술이 적용되기 이전에 가장 많이 사용되던 단어 및 문서를 숫자 형태로 나타내는 가장 간단한 기법이다.
- 이를 활용한 대표적인 문서분류 기법인 NaiveBayes 분류에 대하여 학습한다.

### 2-1. Bag-of-Words Representation
- Constructing the vocabulary containing unique words
![1](https://user-images.githubusercontent.com/53552847/132518863-5c8cd865-fa16-42e6-98df-b50b8e5a923f.PNG)

- Encoding unique words to one-hot vectors
![2](https://user-images.githubusercontent.com/53552847/132518864-9ee1be62-1e40-4222-afe7-2aefa8657903.PNG)
    - one-hot vector의 경우 word embedding과 대비되는 특성이다.
    - 각 단어쌍의 거리가 root(2)로 동일하고 단어간의 유사도인 cosine similarity도 0으로 동일하게 나타내어진다.

- A sentence/document can be represented as the sum of one-hot vectors
![3](https://user-images.githubusercontent.com/53552847/132518867-c1892f6f-f48b-45fd-bd65-d06c43e90780.PNG)
    - 각 문장에서 포함된 word들의 one-hot vector들을 모두 더하여 문장을 vector로 나타낼 수 있고 이를 bag-of-words Vector라고 부른다.

### 2-2. NaiveBayes Classifier for Document Classification
Bayes' Rule Applied to Documents and Classes

- 먼저 분류가 될 수 있는 Category가 c개 있다고 하고 특정한 문서 d가 주어졌을 때, 그 문서가 각 category에 속할 확률분포는 다음과 같다.
![4](https://user-images.githubusercontent.com/53552847/132518868-101387f5-1c23-4182-9983-f177f2379a34.PNG)
- 더불어, P(d)는 d가 고정된 문서라고 할 수 있으므로, P(d)는 상수로 볼 수 있고, argmax operation 상에서는 무시할 수 있는 값이므로 위의 식의 결과처럼 나타낼 수 있다.

![5](https://user-images.githubusercontent.com/53552847/132518874-e3de22a2-1bed-4ce9-8147-46511a06af6f.PNG)
- 위의 그림에서, P(d|c)는 특정 카테고리 c가 고정되었을 때, 문서 d가 나타날 확률이다.
- d는 첫번째 word $$w_1$$부터 마지막 word $$w_n$$까지가 동시에 나타나는 동시 사건으로 나타낼 수 있고, 이 때 각 단어가 등장할 확률 c가 고정되어있을 경우, 각 단어들이 나타날 확률이 서로 독립이라면 위의 식처럼 곱으로 표현할 수 있다.
- 어떤 문서가 주어지기 이전의 각 클래스가 나타날 확률인 P(c)와 특정 클래스가 고정되어 있을 때, 각 word가 나타날 확률들을 추정함으로서 NaiveBayes Classifier에서 필요로하는 Parameter를 모두 추정할 수 있다.
- 다음의 예시를 보며 이해를 도울 수 있다.
![6](https://user-images.githubusercontent.com/53552847/132518875-83720b47-5955-47c3-b333-b730e5ef82a4.PNG)
![7](https://user-images.githubusercontent.com/53552847/132518879-32c78b45-0fcd-47e1-9b71-00e6473543b2.PNG)
![8](https://user-images.githubusercontent.com/53552847/132522531-5586f806-a788-4c98-81c9-356723b639db.PNG)
    - 위의 표에서 각 클래스가 나타날 확률은 cv, nlp 모두 1/2d이다.
    - 각 단어들이 각 class의 document에서 등장할 확률은 위의 표와 같다.
- 위에서 설명한 것과 같이 NaiveBayes Classifier는 class가 3개 이상인 경우에서도 확장하여 사용할 수 있다.
- 위의 일반적인 NaiveBayes Classifier의 경우 단어가 하나라도 포함되어 있지 않게되면 그 Document가 특정 class로 분류될 확률이 0으로 지정되기 때문에 이를 해결하기 위하여 다양한 Regularization과 같은 기법들을 사용하여 진행하기도 한다.
