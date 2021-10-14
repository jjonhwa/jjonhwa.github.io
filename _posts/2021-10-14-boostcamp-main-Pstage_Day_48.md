---
layout: post
title: "Third P-stage 3(Day 48)"
categories: boostcamp
tags: main
comments: true
---
Passage Retrieval에서 Sparse Embedding과 Dense Embedding에 대하여 학습한다.

**부스트 캠프 48일차 학습 요약**
- **행사** : 오피스아워, 멘토링
- **학습** : Passage Retrieval - Sparse Embedding, Passage Retrieval - Dense Embedding
- **피어 세션**

## 목차
- [1. Passage Retrieval - Sparse Embedding](#1-passage-retrieval---sparse-embedding)
- [2. Passage Retrieval - Dense Embedding](#2-passage-retrieval---dense-embedding)
- [3. 해야할 일](#3-해야할-일)

## 1. Passage Retrieval - Sparse Embedding
### 1-1. Introduction to Passage Retrieval
- Passage Retrieval이란?
![29](https://user-images.githubusercontent.com/53552847/137152111-dbe1db60-04f9-43b1-8454-71ce9bc71f64.PNG)
![30](https://user-images.githubusercontent.com/53552847/137152114-7327d106-956e-4488-b526-b45e844adfe5.PNG)
    - 위의 그림처럼, 질문에 맞는 문서를 찾아주는 것을 의미한다.
- Passage Retrieval with MRC
![31](https://user-images.githubusercontent.com/53552847/137152118-055b02f4-ec16-4235-9d74-c386051c68bd.PNG)
    - Open-Domain Question Answering
        - 대규모의 문서 중에서 질문에 대한 답을 찾는 것
    - Passage Retrieval과 MRC를 이어서 2Stages로 문제를 해결할 수 있다.
    - Passage Retrieval에서는 질문에 관련된 혹은 질문에 대한 답을 포함하고 있을 것 같은 지문을 찾아서 MRC에 넘긴다.
    - MRC 모델은 그 지문을 바탕으로 정확한 답변을 하는 과정을 거친다.
- Overview of Passage Retrieval 
![32](https://user-images.githubusercontent.com/53552847/137152122-1163f120-ff9a-47dc-a9b9-91ab678a7baf.PNG)
    - Query와 Passage를 embedding한 뒤, 유사도로 랭킹을 매긴 후, 유사도가 가장 높은 Passage를 선택한다.
    - 유사도를 구하는 방식은 여러가지가 있을 수 있다. (Euclidean distance, dot product, cosine similarity etc)

### 1-2. Passage Embedding and Sparse Embedding
#### Passage Embedding Space
- Passage Embedding의 벡터 공간
- 벡터화된 Passage를 이용하여 Passage 간 유사도 등을 알고리즘을 통해 계산할 수 있다.

#### Sparse Embedding
- Sparse
    - Dense의 반대말
    - 0이 아닌 숫자가 상당히 적게 포함되어있음을 의미한다.
- Bag-of-words(BoW)
    - 문서가 주어졌을 때, 문서를 embedidng space로 mapping하기 위해서 각 문서에 존재하는 단어들을 1과 0(존재하면 1, 존재하지 않으면 0)으로 표현하여 아주 긴 vector로 표현하는 방식이다.
    - 이 경우, 특정 단어가 존재하는지, 존재하지 않은지로 표현하다보니 vector의 길이는 전체 vocabulary의 size와 동일하다.

#### Sparse Embedding 구성 방법
- BoW 구성 방법
    - unigram
        - 각 단어 하나를 한 개로 바라보는 방법
        - 이 경우에는 vector의 크기가 vocabulary size와 같다.
    - n-gram
        - 단어를 n개씩 묶어서 n개를 하나의 단어로 보고 vocabulary를 형성하게 된다.
        - vocab_size는 unigram의 n제곱 형태로 늘어나게 된다. 보통 너무 길어지기 때문에 bigram까지만 활용하는 편이다.
- Term value를 결정하는 방법
    - Term이 document에 등장하는지 (binary)
        - 이는 매우 simple한 방법론으로서 자주 사용하지는 않는다.
    - Term이 몇 번 등장하는지 (term frequency) 등 (e.g. TF-IDF) 

#### Sparse Embedding 특징
- Dimension of embedding vertor 
    - term의 개수와 동일하다. 즉 vocab_size에 의해서 결정된다.
    - 등장하는 단어가 많아질수록 차원이 증가한다.
    - vocabulary의 크기가 커지면 커질수록 vector의 크기가 늘어난다.
    - N-gram term을 고려할 때, n이 커지면 커질수록 기하급수적으로 차원이 증가한다.
- 장점
    - Term Overlap을 정확하게 잡아내야 할 때 유용하다.
    - 검색에 활용할 때, 검색에 활용할 단어가 실제 문서에 들어가 있는지 없는지를 볼 때 정말 유용하다.
    - 질문에 대한 문서를 찾을 때, 질문에 들어있는 단어를 매핑하여 찾을 수 있어 유용하다.
- 단점
    - 의미(semantic)가 비슷하지만 다른 단어인 경우 비교가 불가능하다.
    - 이러한 문제로 인하여, Dense Embedding을 활용하게 된다.

### 1-3. TF-IDF
- TF-IDF
    - Term Frequency - Inverse Document Frequency
    - Term Frequency (TF): 단어의 등장 빈도
    - Inverse Document Frequency (IDF): 단어가 제공하는 정보의 양
        - 단어가 얼마나 자주 등장하는지 혹은 덜 등장하는지를 확인하여 정보량을 나타낸다.
- TF (Term Frequency)
![33](https://user-images.githubusercontent.com/53552847/137152127-87a85ffc-9bd0-4f3d-bc44-ffdc0d8d4173.PNG)
    - 해당 문서 내의 등장 빈도를 의미한다.
    - Raw count: 단순히 개수만 새는 방식. 많이 사용하지는 않는다.
    - Adjusted for doc length: Raw count / num words. TF에서 채택하는 방식이다.
    - Other variants: binary, log normalization 등의 방식도 있다.
- IDF (Inverse Document Frequency)
    - 단어가 제공하는 정보의 양 
![34](https://user-images.githubusercontent.com/53552847/137152129-a830e9ad-8a6e-46d7-96f2-ddaa41f02f11.PNG)
        - Document Frequency(DF) = Term t가 등장한 document의 개수
        - 즉, 여러 document에서 term t가 얼마나 자주 등장했는지를 바탕으로 정보량을 계산한다.
        - 적게 나타나는 정보일수록 많은 정보를 담고 있다고 판단하고 높은 점수를 주고 많은 document에 나타나는 term일수록 낮은 점수를 부여한다.
        - 모든 문서에 등장한 단어의 경우 IDF의 점수는 0점이다.
        - IDF를 TF와 곱해서 최종 점수로 채택한다.
    - 중요한 점
        - IDF의 경우, Document 마다가 아닌 Term 마다 unique하다는 성격이 있다.
        - 문서마다 점수가 다르지 않다.
- Combine TF & IDF 
    - TF-IDF(t, d): TF-IDF for term t in document d
    ![35](https://user-images.githubusercontent.com/53552847/137152131-0b60f7c0-adaf-4e96-9994-3aad2070f796.PNG)
        - document d에 대한 term t의 대한 TF를 계산하고 document d와는 무관하게 term t에 대한 IDF를 계산하여 연산을 진행한다.
        - 'a', 'the'와 같은 관사는 거의 모든 Document에 존재하기 때문에 IDF의 값이 낮아서 Low TF-IDF를 가진다.
        - 자주 등장하지 않는 고유명사의 경우 높은 IDF의 값을 가짐으로서 전체적으로 High TF-IDF를 가진다.

### 1-4. BM25
![36](https://user-images.githubusercontent.com/53552847/137152132-9847d8a8-4915-4407-a3b9-59d0b8b02b54.PNG)
- TF-IDF의 개념을 바탕으로, 문서의 길이까지 고려하여 점수를 매긴다.
- TF 값의 한계를 지정해두어 일정한 범위를 유지하도록 한다.
- 평균적인 문서의 길이보다 더 작은 문서에서 단어가 매칭된 경우, 그 문서에 대해 가중치를 부여한다.
- 실제 검색엔진, 추천 시스템 등에서 아직까지도 많이 사용되는 알고리즘이다.
- 이를 활용할 때, 좀 더 Heuristic하게 사용할 수도 있다.

## 2. Passage Retrieval - Dense Embedding
### 2-1. Introduction to Dense Embedding
#### Passage Embedding
- 구절을 vector로 변환하는 것
- TF-IDF와 같은 Sparse Embedding은 vector의 크기가 vocabulary size에 비례하여 커지고 90% 이상의 element가 0이다.

#### Limitations of Sparse Embedding
- Sparse Embedding의 문제는 차원 수가 매우 크다라는 것이다. 하지만 이 문제는 compressed format으로 어느 정도 극복이 가능하다.
- 가장 큰 문제점은 유사성을 잘 고려하지 못한다는 것이다.
    - 두 단어가 아주 유사한 단어일 지라도 다른 형태의 text로 존재하면 두 단어의 vector space 상 완전히 다른 dimension을 차지한다.
    - 이로 인해, vector space 상에서는 전혀 유사성을 고려할 수 없게 된다.

#### Dense Embedding이란?
![37](https://user-images.githubusercontent.com/53552847/137152133-1a24a6b3-f66e-439f-9f74-b625c3d0774f.PNG)
- Complementary to sparse representations by design
- 더 작은 차원의 고밀도 벡터 (length = 50 ~ 1000)
    - 보통 1000차원을 넘어가는 경우는 거의 없다.
- 각 차원이 특정 term에 대응되지 않는다.
- 대부분의 요소가 non-zero이다.
- 위의 그림에서 왼쪽이 Sparse Embedding을 나타내며, 오른쪽이 Dense Embedding을 나타낸다.

#### Sparse vs Dense
- Sparse
    - Retrieval이 진행될 때, 단어 존재의 유무를 맞출 때는 유용하지만, 실제로 의미론적으로 해석이 용이하지 않다.
    - Dimension이 크기 때문에 활용할 수 있는 알고리즘의 한계가 있다.
    - 중요한 term들이 정확히 일치해야 하는 경우 성능이 뛰어나다.
    - Embedding이 구축되고 나서는 추가적인 학습이 불가능하다.
- Dense
    - 의미가 같은 다른 단어들 같은 경우들도 잘 채택할 수 있다.
    - Dimension이 보통 50 ~ 1000으로, 훨씬 더 많은 종류의 알고리즘을 활용할 수 있따.
    - 단어의 유사송 혹은 맥락을 파악해야하는 경우, 성능이 뛰어나다.
    - 학습을 통해 임베딩을 만들며, 추가적인 학습 또한 가능하다.
- 실제 현업에서 Sparse Embedding이 사용되긴 하지만, Sparse Embedding만으로 많은 것을 하기는 쉽지않다.
- 실제로는 Sparse Embedding을 쓰는 동시에, Dense Embedding 사용하거나, 또는 Dense Embedding만을 활용하여 Retrieval을 구축하는 것을 일반적으로 추천한다.
- 최근 사전 학습 모델의 등장, 검색기술의 발전 등으로 인해 Dense Embedding을 활발히 이용하는 추세이다.

#### Overview of Passage Retrieval with Dense Embedding
![38](https://user-images.githubusercontent.com/53552847/137152135-b9844dd9-8dad-4400-8096-ab98e5b5d850.PNG)
- 위의 그림과 같이 Dense Embedding을 만들게 된다.
- 왼쪽에는 Question에 대응하는 BERT Encoder가 존재한다.
    - Question이 들어왔을 때, BERT Question에 해당하는 encoder가 sentence encoding을 하여 [CLS] token에 해당하는 embedding vector를 출력하고 이를 $$h_q$$라고 지칭한다.
- 오른쪽에는 Passage에 대응하는 BERT Encoder가 존재한다.
    - Passage가 들어왔을 때, BERT Passage에 해당하는 encoder가 sentence encoding을 하여 [CLS] token에 해당하는 embedding vector를 출력하고 이를 $$h_b$$라고 지칭한다.
- 이 때 핵심은, $$h_q$$와 $$h_b$$가 같은 size를 가진다는 것이다. 같은 size일 경우에 유사도를 구할 수 있다.
- 가장 일반적으로 사용하는 유사도 연산은 dot product가 있고 위의 그림에서도 dot product를 활용하여 연산을 진행한다.
- 위의 그림의 경우, 특정 Passage와 Question 사이의 유사도를 구한 것이며, 모든 passage에 대한 Question과의 유사도를 구해서 가장 높은 유사도를 가지는 Passage를 찾아서 이를 Reader에 넣어주도록 한다.
- BERT encoder를 추가적으로 학습해야하며, Question BERT와 Passage BERT는 서로 다를 수도 있고 같은 모델을 활용할 수도 있다. 다만, architecture는 동일하다.

### 2-2. Training Dense Encoder
#### What can be Dense Encoder
- BERT와 같은 Pre-trained Language Model (PLM)이 자주 사용된다. 
- BERT만이 옵션은 아니며 다양한 PLM을 활용하여 fine-tuning을 하여 활용할 수 있다.
- MRC는 passage, question 둘 다 넣어주는 반면, Training Dense Encoder의 경우, Passage와 Question을 각가 넣어서 각각의 Embedding을 구하고 싶기 때문에 독립적으로 넣어주게 된다.
- [CLS] token의 최종 embedding을 output한다.
- Passage, Question BERT를 각각 다른 모델을 사용하는 것이 좋을지 혹은 하나로 통합하여 사용하는 것이 좋을 지는 Design에 따라 다르고, 경우에 따라서는 하나로 통합하여 사용하는 것이 더 성능이 좋을수도 있다.

#### Dense Encoder 학습 목표와 학습 데이터
- 학습 목표
    - 연관된 Question과 Passage Dense Embedding 간의 거리를 좁히는 것
    - 혹은 유사도를 높이는 것. High Similarity
- Challenge
    - 연관된 Question / Passage를 어떻게 찾을 것인가?
    - 기존 MRC 데이터 셋을 활용하여 학습한다.
- Negative Sampling 
![39](https://user-images.githubusercontent.com/53552847/137152139-ee0d2135-2a07-4548-9827-5365fe2df615.PNG)
    - 연관된 Question과 Passage 간의 Dense Embedding 거리를 좁히는 것
    - 연관되지 않은 Question과 Passage 간의 Dense Embedding 거리를 넓히는 것
- Choosing Negative Examples
    - Corpus 내에서 랜덤하게 뽑기
    - 좀 더 헷갈리는 Negative 샘플 뽑기. (ex, 높은 TF-IDF score를 가지지만 답을 포함하지 않는 샘플)
    - 실제로 바로 위의 방법론을 활용하여 정확성을 올린 케이스가 많이 존재한다.
- Objective Function
![40](https://user-images.githubusercontent.com/53552847/137152141-78a8c8bd-205e-4c99-a065-c991c79954c2.PNG)
    - Positive Passage에 대한 Negative Log Likelihood(NLL) Loss를 사용한다.
    - 결국, Possitive Passage의 score를 확률화하기 위하여 Positive Passage와 Question 간의 Similarity score(= Real Number, 높을 수록 유사도가 높다.)와 Negative Passage와 Question 간의 Similarity score를 가져와서 softmax를 하여 softmax한 값의 확률값을 NLL에 적용하여 학습하는 방법을 택한다.
    - 학습 Corpus를 Question, Positive Passage, Negative Passage로 나눈다고 하면, 실제로 Loss를 compete하는 방법은 negative, log 그리고 softmax를 적용한다.
    - Softmax의 경우, 위의 그림에서 볼 수 있듯이, 분모에 Negative score + Positive score가 있고 분자에 positive score를 주어 계산한다.

#### Evaluation Metric for Dense Encoder
- Top-k Retrieval Accuracy
![41](https://user-images.githubusercontent.com/53552847/137152144-c5748a0e-b094-42b1-a049-8602a1b94a32.PNG)
    - Retrieve된 Passage 중 답을 포함하는 Passage의 비율이다.
    - Ground-truth passage를 알 수 있음으로 이를 바탕으로 점수를 확인한다.

### 1-3. Passage Retrieval with Dense Encoder
#### From Dense encoding to retrieval
![42](https://user-images.githubusercontent.com/53552847/137152150-d74393ec-1a6a-4598-9e67-84326100070a.PNG)
- Inference: Passage와 Query를 각각 Embedding한 후, Query로부터 가까운 순서대로 Passage의 순위를 매긴다.

#### From retrieval to open-domain question answering
![43](https://user-images.githubusercontent.com/53552847/137152153-99518b93-986d-48aa-8a22-42b23e6ff368.PNG)
- Retriever를 통해 찾아낸 Passage를 활용, MRC(Machine Reading Comprehension)

#### How to make better dense encoding
- 학습 방법 개선 (e.g. DPR) -> Reference Paper 참조
- 인코더 모델 개선 (BERT보다 큰, 정확한 Pre-trained 모델)
- 데이터 개선 (더 많은 데이터, 전처리 등)
- MRC 모델의 성능향상도 중요하지만, 얼마나 올바른 문서를 찾아오는 가 역시 중요하다.

## 3. 해야할 일
-  Question BERT [CLS] token과 Passage BERT [CLS] token 사이의 유사도를 구할 때, **유사도를 구하는 다양한 방법론** 생각해보기
- Question BERT, Passage BERT에서 BERT가 아닌 **다양한 PLM 모델을 활용**하여 fine-tuning 할 수 있다.
- 유사도를 구할 때, [CLS] token에 대한 embedding vector를을 사용하는데, **다른 유의미한 token**은 없을까?
- Negative Sampling을 진행할 때, **Negative Example를 선택하는 방법론**을 여러가지 적용해보기 (ex, 높은 TF-IDF score를 가지지만 답을 포함하지 않은 샘플 등) -> 실제로 정확도가 많이 올랐다고 한다.
- Dense Encoding을 할 때, 학습 데이터는 어떤 것을 활용해야 하는가? -> 공개된 KorQuAD Dataset?! 혹은 Competition에서 제공된 dataset?!
- NLL을 사용하면 자동으로 Ground Truth와는 더 가까워지도록, 그리고 답이 아닌 Passage들과는 멀어지도록 학습되는가?
