---
layout: post
title: "Third P-stage 4(Day 49)"
categories: boostcamp
tags: main
comments: true
---
Retriever의 FAISS, Retrieval와 MRC를 연결하는 방법에 대하여 학습한다.

**부스트 캠프 49일차 학습 요약**
- **행사** : 마스터 클래스, 멘토링
- **학습** : Scaling up with FAISS, Linking MRC and Retrieval
- **피어 세션**

## 목차
- [1. Scaling up with FAISS](#1-Scaling-up-with-faiss)
- [2. Linking MRC and Retrieval](#2-linking-mrc-and-retrieval)
- [3. 해야할 일](#3-해야할-일)

## 1. Scaling up with FAISS
### 1-1. Passage Retrieval and Similarity Search
#### Passage Retrieval
- Question의 경우, 질문이 들어올 때마다 Encoding을 진행해주어야 한다.
- Passage의 경우, 미리 Passage를 확보한 상태라면, offline으로 전부 연산을 진행한 후 저장을 해놓고 새로운 질문이 들어올 떄마다 기존 passage들과의 유사도를 계산하여 가장 높은 유사도를 가진 passage 내보내는 방식을 택한다.
- Inference의 경우, Passage와 Query를 각각 Embedding한 후, Query로부터 거리가 가까운 순서대로 Passage에 순위를 매긴다.
- 이 때, Passage의 개수가 늘어날수록 거리를 계산해야할 Passage가 많아진다는 것인데, 기본적으로 Query와 Passage 사이의 유사도를 dot product를 활용하여 연산을 진행하는데 dot product 조차도 passage의 개수가 많아져 dimension이 늘어날 경우 부담스러운 연산을 수행하게 된다.
- Passage의 개수가 많아질 경우 어떻게 효율적으로 가장 가까운 Passage를 구할 수 있을 지에 대하여 고민해본다.

#### MIPS (Maximum Inner Product Search)
- Question Embedding과 모든 Passage Embedding 사이의 Inner Product연산을 통해서 가장 큰 값을 가지는 Passage를 가장 유사한 Passage로 판단하고 Reader로 보내준다.
- Nears Neighbor Search(L2 Norm을 활용한 Euclidean Distance)보다는 Inner Product Search를 많이 사용한다.
- Nears Neighbor Search의 경우, 개념적으로 설명하기가 쉽기 때문에, 설명을 할 때 많이 인용해서 설명한다.
- 실질적으로 학습할 때, 효율성 차원으로 봤을 때, Dot Product 연산을 통해 가장 높은 값을 찾이 조금 더 수월하다.
- Challenges
    - 실제로 검색해야할 데이터는 훨씬 방대하므로 모든 문서를 일일히 보면서 검색할 수는 없다.
    - 위키피디아 - 약 5백만개
    - Embedding size가 수십억, 조 단위까지 커질 수 있다.

#### Trade-off of similarity search
- Similarity Search: Passage의 개수가 많아질 경우 효율적으로 Question에 가장 유사한 Passage를 검색하는 것
- 방대한 문서를 짧은 시간 내에 훑고 가장 가까운 passage embedding vector를 찾는 알고리즘이 상당히 중요하다.
- 시간과 정확도 사이의 trade off를 고려하여 가장 좋은 문서를 탐색할 수 있어야 한다.
![44](https://user-images.githubusercontent.com/53552847/137234028-6f53d719-265c-43ec-ab79-946a95d9c61d.PNG)
- Search Speed
    - Query 당 유사한 벡터 k개를 찾는데 얼마나 걸리는가?
        - 가지고 있는 벡터량(passage의 양)이 클수록 오래걸린다.
    - Inference time에서 Prunning과 같은 Mechanism을 활용해서 Approximation을 더 많이 하는 대신 경우에 따라서는 정확성을 조금 잃더라도 필요한 속도를 확보할 수 있다.
- Memory Usage
    - 벡터를 사용할 때, 어디서 가져올 것인가?
        - RAM에 모두 올려둘 수 있으면 빠르지만 많은 RAN 용량을 요구한다.
        - 디스크에서 계속 불러와야한다면 속도가 느려진다.
- Accuracy
    - brute-force 검색 겨로가와 얼마나 비슷한지?
        - 속도를 증가시키려면 정확도를 희생해야하는 경우가 많다.
    - 어떻게하면 검색을 좀 더 효율적으로 할 것인가?

#### Trad-off of search speed and accurayc benchmark
![45](https://user-images.githubusercontent.com/53552847/137234029-a43d743a-c480-4600-8102-56c6cc0a1c1f.PNG)
- 속도와 재현율의 관계는 trade off 관계가 잆음을 위의 그래프로부터 확인할 수 있다.
- 더 정확한 검색을 하기 위해서는 더 오랜 시간이 소모된다.

#### Increasing search space by bigger corpus
![46](https://user-images.githubusercontent.com/53552847/137234030-4ccb5ed3-4f7d-44d2-b1b4-f416659d6e97.PNG)
- 코퍼스의 크기가 커질수록 탐색 공간이 커지고 검색이 어려워진다.
- 저장해 둘 Memory Space가 많이 요구된다.
- Sparse Embedding의 경우 이 문제가 훨씬 심하다.

### 1-2. Approximating Similarity Search
#### Compression - Scalar Quantization (SQ)
![47](https://user-images.githubusercontent.com/53552847/137234031-83320bca-c2bc-4ba7-95f6-7280eae80854.PNG)
- Passage Embedding vector를 압축하여, 하나의 Embedding vector가 적은 용량을 차지하도록 압축량을 늘려서 메모리를 줄이는 방식. 하지만 정보의 손실이 따른다.
- 보통 정보를 저장할 때, float32라는 4byte 체계를 활용한다. 하지만, 실제 inner product search를 할 때는 4byte가 필요한 경우가 드물고, 1byte로 approximate 한 후 저장을 한다 하더라도 상당히 정확한 경우가 많다.
- 이를 SQ라고 하며, 이름 그대로 각각 수치르 Quantize를 해서 Quantize한 값에 대하여 용량을 줄일 수 있도록 해주는 압축 알고리즘으로 이해할 수 있다.

#### Pruning- Inverted File(IVF)
- Pruning
    - Search space를 줄여 search 속도를 개선한다.
    - Dataset의 subset만을 방문하여 subset에 속하는 passage에 대해서만 연산을 진행한다.
    - Clustering을 진행한 후, Invereted file을 활용한 search 방법
- Clustering
    - 전체 vector space를 k개의 cluster로 나눈다. (K-means Clustering) 
- 가장 가까운 k개의 군집을 선택하고 선택된 군집에 대해서는 Exhaustive Search를 진행한다.
- IVF (Inverted File)
![48](https://user-images.githubusercontent.com/53552847/137234032-bcc75a5d-2ca5-4042-8c41-517f645cb68d.PNG)
    - vector의 index = inverted list structure
    - 각 cluster의 centroid id와 해당 cluster의 vector들이 연결되어 있는 형태
- Searching with clustering and IVF
![49](https://user-images.githubusercontent.com/53552847/137234034-9309de94-72f3-4b6b-a8be-ae62e4e6453b.PNG)
    - 주어진 Query vector에 대해 근접한 centroid 벡터를 찾는다.
    - 찾은 Cluster의 Inverted list 내 vector들에 대하여 Search를 수행한다.
    - Search space를 실제로 빠른 시간 내에 비약적으로 줄일 수 있다.

### 1-3. Introduction to FAISS
- Fast Approximate를 위한 Library이다.
- 실제로 Large Scale에 상당히 특화되어 있어, scale up할 때 상당히 용이하게 사용될 수 있다.
- Backbone은 C++로 되어 있는데, wrapping은 python으로 구성되어 있어서 python에서도 유용하게 사용할 수 있따.
- FAISS
    - Library for efficient similarity search
    - indexing을 도와준다고 이해할 수 있다.

#### Passage Retrieval with FAISS
![50](https://user-images.githubusercontent.com/53552847/137234035-e4436257-2e29-4112-adb3-8bd6115d2010.PNG)
- 학습
    - FAISS를 활용하기 위해서 Prunning에서 볼 수 있듯이 cluster를 확보해야하는데 완전히 random하게 cluster를 정하는 것은 비효율적이다.
    - 실제로 cluster들은 data point들의 분포를 보고 적절한 cluster를 지정해야하므로, 이를 지정하기 위한 학습 데이터가 필요하다.
    - 더불어, SQ 과정에서도 큰 float number를 integer로 압축시키는 것으로 볼 수 있는데, 이렇게 되면 float number의 max, min이 얼마인지 알고, 얼마로 scale하고 얼마나 upset할 것인지 파악할 필요하고 있고 이로 인해 FAISS를 활용해 index building 할 때 학습이 필요하다.
    - 학습단계에서 Clustering과 SQ를 하는 비율과 upset을 계산하게 된다.
- 학습 단계에서 Cluster와 SQ8이 정의되고 dataset을 투입하여 실제로 cluster 내의 SQ8형태로 압축된 벡터들을 Question embedding vector와 연산한다.
- dataset을 활용하여 FAISS를 학습시키고 dataset을 FAISS에 adding하여 Question embedding과 유사도를 구할 passage embedding을 추출한다.
- 다만, 많은 경우에는 학습할 데이터와 add하는 데이터를 따로 하지 않고, add할 데이터가 너무 많을 경우에 전부를 학습하는 것은 비효율적이기에 add dataset의 일부를 sampliing해서 학습 데이터로 활용하고, 정말 커질 경우에는 1/40 정도만 sampling해서 쓰는 경우가 있다.

#### Search based on FAISS index
![51](https://user-images.githubusercontent.com/53552847/137234036-1bd987bf-5287-4a54-bef6-96d96a439428.PNG)
- 학습된 FAISS에 passage embedding vector를 넣어주고, 마지막으로 Query vector를 넣어서 Search를 진행한다.
- `nprobe` option은 Query vector와 가까운 cluster를 몇 개를 방문할 것인지 정한다.
- cluster 내의 벡터들과 Query vector를 Exhaustive Search를 진행하여 top-k개의 가장 가까운 passage embedding vector를 뽑는다.
- FAISS는 아주 일반적인 Library 이므로 어떤 종류의 벡터이든지 그 유사성을 구할 수 있다.
- 예를 들어, 가장 가까운 10개의 cluster를 선택한 다음, 이 cluster 내의 문서들 중 SQ8 base로 search를 하고 top-k개를 search result로 내보내주게 되는 것이다.

### 1-4. Scaling up with FAISS
#### FAISS Basics
- brute-forced로 모든 vector와 query를 비교하는 가장 단순한 인덱스를 만든다.
- SQ를 진행할 때만 train 한다.

#### IVF with FAISS
- IVF 인덱스 만들기
- Clustering을 통해서 가까운 Cluster 내 vector과만 비교
- 빠른 검색이 가능하다.
- clsuter 내에서는 여전히 전체 벡터와 거리를 비교한다.

#### IVF-PQ with FAISS
- 벡터 압축 기법 (PQ) 활용
- 전체 벡터를 저장하지 않고 압축된 벡터만 저장
- 메모리 사용량을 줄일 수 있다.

#### Using GPU with FAISS
- GPU의 빠른 연산 속도를 활용할 수 있다.
    - 거리 계산을 위한 행렬곱 등에서 유리
- 다만, GPU 메모리 제한이나 메모리 random access 시간이 느린 것이 단점이다.
- 여러 GPU를 활용하여 연산 속도를 한층 더 높일 수 있다.

## 2. Linking MRC and Retrieval
### 2-1. Introduction to ODQA
#### Linking MRC & Retrieval (= ODQA)
- MRC: 지문이 주어진 상황에서 질의응답
- ODQA: 지문이 주어지지 않고 넓은 의미로 Wikipedia 혹은 웹전체가 주어진 후 이에 기반하여 질의응답한다.
- Modern Search Engines: 연관문서 뿐만 아니라 질문의 답을 같이 제공해준다.

#### History of ODQA
- Text Retrieval Conference (TREC) - QA Tracks(1999-2007)
    - 연관문서만 반환하는 information retrieval(IR)에서 한 걸음 더 나아가서 short answer with support 형태가 목표이다.
- Question Processing + Passage Retrieval + Answer Processing 
- Question Processing
    - 어떻게 하면 질문을 잘 이해할 수 있을까?
    - Query Formation: 질문으로부터 키워드를 선택해서 Answer type selection하는 방식이 유일했다. (질문에 대한 답변이 Answer type별로 미리 정해져 있고 이들 중에서 선택하여 답한다.)
-  Passage Retrieval
    - 기존의 IR 방법을 활용해서 연관된 Document를 뽑고, Passage 단위로 자른 후 선별한다.
    - Named entity, Passage 내의 Question 단어의 개수 등과 같은 hand-crafted features를 활용했다.
- Answer Processing
    - Hand-crafted features와 Heuristic을 활용한 Classifier
    - 주어진 Question과 선별된 Passage들 내에서 답을 선택한다.

### 2-2. Retriever-Reader Approach
#### Retriever
- 데이터베이스에서 관련 있는 문서를 검색한다.
- 입력: 문서셋, 질문
- 출력: 관련성 높은 문서
- 학습 단계
    - TF-IDF, BM25(Sparse Embedding) : 학습 없다.
    - Dense Embedding : 학습 있다.

#### Reader
- 검색된 문서에서 질문에 해당하는 답을 찾아낸다.
- 입력: Retrieved된 문서, 질문
- 출력: 답변
- 학습 단계
    - SQuAD와 같은 MRC dataset으로 학습한다.
    - 학습 데이터를 추가하기 위해서 Distant Supervision을 활용한다.

#### Distant Supervision
- 질문-답변만 있는 데이터셋(CurratedTREC, WebQuestions, WikiMovies)에서 MRC 학습 데이터 만들기. Supporting Document가 필요하다.
- 위키피디아에서 Retriever를 이용해 관련성 높은 문서를 검색
- 너무 짧거나 긴 문서, 질문의 고유명사를 포함하지 않은 부적합한 문서는 제거한다.
- answer가 extract match로 들어있지 않은 문서는 제거한다.
- 남은 문서 중 질문과(사용 단어 기준) 연관성이 가장 높은 단락을 supporting evidence로 사용한다.

#### Inference
- Retriever가 질문과 가장 관련성 높은 5개의 문서 출력
- Reader가 5개의 문서를 읽고 답변 예측
- Reader가 예측한 답변 중 가장 score가 높은 것을 최종 답으로 사용한다.

### 2-3. Issues and Recent Approaches
#### Different granularities of text at indexing time
![52](https://user-images.githubusercontent.com/53552847/137234038-55bdfcdd-f6ee-4d5a-a3cf-863851639c3d.PNG)
- 위키피디아에서 각 Passage의 단위를 문서, 단락 혹은 문장 단위 중 어떻게 가져올 것인지 정의해야 한다.
    - Article의 경우 5.08 million, Paragraphe는 29.5 million, Sentence는 75.9 million 가량이 존재한다.
- Retriever 단계에서 몇 개(top-k)의 문서를 넘길지 정의해야 한다.
    - 이 때, granularity에 따라서 넘기는 문서의 개수 k가 달라진다.
    - Article의 경우 5개 정도면 충분할 수 있지만, Paragraph는 더 많아야 하고, Sentence는 그보다 더 많아야 한다.
    - K를 늘리면 늘릴수록 성능이 올라가는 경우가 있지만 항상 그러는 것은 아니다.
    - 이런 k가 hyperparameter로서 잘 tuning해야한다.

#### Single-Passage Training vs Multi-Passage Training
- Single Passage
    - 현재 우리는 k개의 passage들을 reader가 각각 확인하고 특정 answer span에 대한 예측 점수를 나타낸다.
    - 이 중 가장 높은 점수를 가진 answer span을 선택하도록 한다.
    - 하지만, 이 경우, retrieved passage들에 대한 직접적인 비교라고 볼 수 없다.
- Multi Passage
    - retrieved passages 전체를 하나의 passage로 취급하고, reader 모델이 그 안에서 answer span 하나를 찾도록 한다.
    - 이 경우, 문서가 너무 길어지므로 GPU에 더 많은 메모리를 할당해야하고 처리해야하는 연산량이 많아진다.

#### Importance of each passage
![53](https://user-images.githubusercontent.com/53552847/137234039-ed71d539-3ddf-48a1-9705-54619e7f2524.PNG)
- Retriever 모델에서 추출된 top-k passage들의 retrieval score를 reader에 전달한다.
- Retriever에서 문서를 선택할 때, 고려한 점수가 있는데, 선택된 문서만 그대로 넘겨주게 되면 Reader 에서는 그러한 점수에 대한 고려가 전혀 없이 최종 답안을 만든다.
    - 경우에 따라서는 top-k passage들의 score를 reader score에 함께 넘겨줌으로서 최종 answer를 고를 때, passage retriever score까지 함께 학습할 수 있다.
    - 이 경우에 성능이 더 높게 나온다고 한다.

## 3. 해야할 일
- brute-force 검색이란?
- cluster학습은 어떤식으로 진행되는가?
- PQ/SQ에 대하여 자세히 알아보기
- Single-passage / Multi-passage training에 대하여 자세히 알아보기

