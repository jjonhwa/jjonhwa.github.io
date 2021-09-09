---
layout: post
title: "NLP Ustage Day 1-2 (Day 25)"
categories: boostcamp
tags: main
comments: true
---
Word Embedding에 대하여 학습한다.

**부스트 캠프 25일차 학습 요약**
- **행사** : MeetUP(w. 피어), MeetUP(부캠에서 살아남기)
- **학습** : Bag-of-Words, Word Embedding
- **피어 세션**

## 목차
- [1. Word Embedding](#1-word-embedding)
- [2. 피어 세션](#2-피어-세션)
- [3. 25일차 후기](#3-25일차-후기)
- [4. 해야할 일](#4-해야할-일)

## 1. Word Embedding
### 1-1. What is Word Embedding?
- Express a word as vector
- Word Embedding은 그 자체가 ML/DL 기술로서 어떤 텍스트 Dataset을 학습데이터로 주게된다.
- 또한, 좌표공간의 차원 수를 사전에 미리 정의해서 그 입력으로 Word Embedidng을 주게되면, Word Embedding의 학습이 완료된 이후, 해당하는 좌표공간 상에서 이 학습데이터에서 나타낸 각 단어의 최적 좌표값 혹은 그에 해당하는 vector 표현형을 출력으로 내어준다.
- Word Embedding의 기본 아이디어는 비슷한 의미를 지니는 단어가 좌표공간상에 비슷한 위치의 점으로 매핑되도록 함으로서 단어들의 의미상 유사도를 잘 반영한 벡터 표현을 다양한 자연어 처리 알고리즘에게 제공해주는 역할을 한다.
- Word Embedding을 통해 의미를 잘 반양한 vector를 입력으로 받음으로서, 자연어 처리 task를 수행할 때 보다 쉽게 성능을 올릴 수 있는 여건을 제공할 수 있다.

### 1-2. Word2Vec
- An algorithm for training vector representation of a word from context words
- Assumption : words in similar context will have similar meanings
- 즉, 비슷한 의미를 가지는 단어가 좌표공간 상에서 가까운 위치로 매핑되도록 하기 위해 Word2Vec 알고리즘은 같은 문장에서 나타난 인접한 단어들 간의 의미가 비슷할 것이라는 가정을 사용한다.
- 따라서 Word2Vec에서는 각 단어가 주변에 등장하는 단어들을 통해 그 의미를 알 수 있다는 사실을 착안하고, 주어진 학습데이터를 바탕으로 Target word 주변에 나타나는 단어들의 확률분포를 예측하게 된다.
![9](https://user-images.githubusercontent.com/53552847/132601362-053c6d94-e458-4bbe-82f9-1c5ca2053c3a.PNG)
- 타겟 단어를 입력으로 주고, 주변 단어를 숨긴채 이를 예측하는 방식으로 Word2Vec 모델의 학습이 진행된다.
![10](https://user-images.githubusercontent.com/53552847/132601365-09d446fe-45b4-4f74-bed2-be1f3946262e.PNG)

### 1-3. Word2Vec Algorithm
학습데이터가 'I study math' 하나라고 가정하고 다음의 설명을 진행한다.

![11](https://user-images.githubusercontent.com/53552847/132601367-a2879f5d-6aec-44e2-a701-dbaf1516c0c0.PNG)
#### 학습데이터 구성
- 학습데이터를 word별로 tokenization을 수행한다.
- Unique한 단어들만 모아서 사전을 구축한다.
- 사전의 각 단어들은 사전의 Size 만큼의 dimension을 가진 One-hot Vector의 형태로 나타내어진다.
- Sliding Window 기법을 적용하여 어떤 한 단어를 중심으로 앞뒤에 나타난 Word 각각과 입출력 단어 쌍을 구성한다. 예를 들어, window size가 2라면, (I, study), (study, I), (study, math)를 구성하게된다.
- 위의 방식으로 통해, 학습데이터에 대해서 각 문장별로 sliding window를 적용하고 중심단어와 주변단어 각각을 단어쌍으로 구성함으로서 Word2Vec의 학습데이터를 구성할 수 있다.

#### Network 구성
- 예측 task를 수행하는 2layer Neural Network를 구성한다.
- Hidden layer의 노드 수는 사용자가 입력하는 Hyper Parameter로 작동한다. 즉, Hidden layer의 노드 수는 Word Embedding을 수행하는 어떤 좌표공간의 차원수로 이해할 수 있다.
- 마지막에 Softmax Layer를 통과해줌으로서 n-dim vector가 특정한 확률분포값을 가지도록 바꿔주고 변환된 확률분포가 ground-truth layer와 동일한 형태를 가질 수 있도록 softmax loss를 활용하여 학습한다.

#### Embedding Layer
- DeepLearning Programming 과정 중, One-hot Vector와 첫번째 선형변환 matrix와 곱해지는 과정을 embedding layer라고 한다.
- 실제로 이 과정에서는, 행렬곱을 수행하지는 않고 one-hot vector의 1이 존재하는 인덱스에 대하여 W의 column vector와 내적을 계산하여 연산을 진행한다.

### 1-4. Property of Word2Vec
- The word vector or the relationship between vector points in space, represents the relationship between the words
![12](https://user-images.githubusercontent.com/53552847/132601368-c671e251-295c-47ba-b112-a54c56945232.PNG)
- 위의 그림과 같이 Word2Vec이 학습한 흥미로운 의미론적 관계들을 확인할 수 있다.

#### Word Intrusion Detection task
- 여러 단어들이 주어져 있을 때, 이 중 나머지 단어들과 의미가 가장 상이한 단어를 찾아내는 task
- Word2Vec이 학습한 각 word들의 Embedding Vector들을 활용해서 쉽게 해결할 수 있다.
- 각 단어별로 각각의 단어들과의 Uclidean Distance를 계산한 후 평균을 취하면 그 값이 단어별로 다른 단어들과 이루는 평균 거리로 이해할 수 있는데, 이 때 평균 거리가 가장 큰 단어를 선택하면 그 단어가 주어진 단어들 중 의미가 가장 상이한 단어라고 할 수 있다.

### 1-5 Another
- https://ronxin.github.io/wevi -> Word2Vec의 또 다른 예시이다.
- Word2Vec은 2layer neural network로서 가중치 행렬 W1, W2가 등장하는데 이들로 부터 이 때, W1의 벡터표현값과 W2의 벡터표현값이 유사할 경우 내적할 때 더욱 큰 값을 가지게되는데 이 때 두 단어의 의미가 유사하다고 생각할 수 있다.
- 또한 각 가중치 행렬에서 벡터표현형이 유사한 값을 보이는 단어들은 비슷한 단어라고 판단을 할 수 있고 W1, W2 중에서 무엇을 Word Embedding의 output으로 사용할지에 대해서는 어떤 것을 사용해도 상관없으나 통상적으로 W1을 Word Embedding의 output으로 사용한다.
- http://w.elnn.kr/search -> Word2Vec의 한글버전 데모

### 1-6. GloVe
- 각 입력 및 출력 단어 쌍들에 대하여 학습데이터에서의 두 단어가 한 윈도우 내에서 동시에 총 몇 번 등장했는지에 대하여 사전에 미리 계산을 한다.
- 아래의 수식에서 볼 수 있듯이, 입력 word의 embedding vector $$u_i$$, 출력 word의 embedding vector $$v_j$$간의 내적값이, 두 단어가 한 윈도우 내에 동시에 총 몇 번 나타났는지에 대한 값 $$P_{ij}$$ 의 log를 취한 값과 정확하게 fitting 될 수 있도록 새로운 형태의 loss function을 사용한다.
![13](https://user-images.githubusercontent.com/53552847/132601370-2cf85b7c-fb85-4082-b7d2-e3fd4a219f7b.PNG)
- 이러한 점에서 GloVe와 Word2Vec의 차이가 존재한다.
- Word2Vec의 경우, 특정한 입출력 단어 쌍이 자주 등장하는 경우, 이런 단어가 여러번에 걸쳐 학습됨으로서 내적값이 커지는 학습방식이었다면, GloVe는 어떤 단어쌍이 동시에 등장하는 횟수를 미리 계산하고 이에 대한 log를 취한 값을 두 단어의 내적값의 ground truth로서 사용해서 학습을 진행한다.
- 즉, 이러한 이유로 인해 GloVe는 Word2Vec에 비해 중복되는 계산을 줄여주고, 더 빠르고 보다 적은 데이터에 대해서도 잘 동작한다는 특성을 보인다.
- 또한, 선형대수의 관점에서 추천시스템 등에 자주 활용되는 알고리즘인 co-occurrent matrix의 어떤 row-rank matrix factorization의 task로도 이해할 수 있다.
- Word2Vec과 GloVe 두 모델 모두 주어진 학습데이터, 텍스트 데이터에 기반해서 Word Embedding을 학습하는 동일한 역할을 수행하는 알고리즘이고 실제 다양한  task에 적용했을 때, 성능도 두 알고리즘이 비슷하다는 것을 알아두자.

## 2. 피어 세션
### 2-1. Spacy Debugging
- 로컬에서 spacy 패키지를 사용할 경우 다음과 같이 진행한다.
    - en 모델은 spacy version 3.0 이후로 depreciated 됬으며, 모델이 세분화 된 것 같다.
    - en_core_web_sm을 사용했으며 이는 순서대로 'en(영어)', 'core(사용목적)', 'web(훈련 데이터셋의 출처)', 'small(모델 크기)'를 의미한다.
    ```
    !pip install spacy
    !python3 -m spacy download en_core_web_sm
    ```
    - 위의 코드를 통해 사용할 수 있다.
- Mac에서는 Python 2.7이 python3에 내장되어있어 `!python3`라고 지정했지만 windows의 경우 `!python`이라고 지정해도 alias로 지정되는 것으로 보인다.

### 2-2. Word2vec 부분 개념 정리
- 알고리즘 동작하는 부분이 이해하기 힘들었다.
- Softmax Matrix Multiplication 부분이 잘 이해가 안 됐다
    - 간단! W1과 W2가 학습 대상
    - W1의 column dimension = hidden dimension. 정보량을 결정
    - x는 단어에 대한 벡터 표현
    - W1, W2를 sliding window로 구한 word tuple의 ground truth 값에 대해 훈련시키는 것
    ![https://i.imgur.com/qBTJula.png](https://i.imgur.com/qBTJula.png)
(참고자료: [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/))

## 4. 25일차 후기
지난 주의 첫번째 P-stage를 마치고 다시 돌아온 U-stage이다! 지난 2주간의 삽질, 실험 등에 지쳐서인지 U-stage가 반가웠고, 가르쳐 주는데로 따라가고 이에 대해서 좀 더 조사하고 알아가는게 혼자서 찾아보고 공부하는 것에 비해 훨씬 편하다는 생각이 들었다.

당연히 U-stage의 난이도가 낮거나 쉬운 것만은 아니지만 이러한 지식을 제때 쌓지 못하고 P-stage에 넘어가게 되면 얼마나 고생하는 지 맛보았기에 U-stage를 조금은 더 즐기며 공부할 수 있는 것 같고 앞으로 좀 더 노력해서 P-stage에서도 도움을 줄 수 있는 누군가가 되기 위해 달려나가야겠다.

## 5. 해야할 일
- Word2Vec과 GloVe 알고리즘이 가지고 있는 단점은 무엇인가?
