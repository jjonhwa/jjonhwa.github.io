---
layout: post
title: "Ustage Day 6"
categories: boostcamp
tags: main
comments: true
---
Neural Network와 딥러닝 기본 용어에 대한 설명을 한다.

**부스트 캠프 6일차 학습 요약**
- **학습** : 딥러닝 기본 용어 설명, Neural Network, 필수 과제
- **피어 세션**

## 목차
- [1. 딥러닝 기본 용어 설명](#1-딥러닝-기본-용어-설명)
- [2. Historical Review](#2-historical-review)
- [3. Neural Network](#3-neural-network)
- [4. 피어 세션](#4-피어-세션)
- [5. 6일차 후기](#5-6일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. 딥러닝 기본 용어 설명
### 1-1. Good Deep Learner가 되려면 어떻게 해야하는가?
- Implementation Skills (구현 실력)
- Math Skills (Linear Algebra, Probability)
- Knowing a lot of recent Papers(최근에 나온 많은 연구 결과들)

### 1-2. Key Components of Deep Learning
- Data : 모델을 훈련시키기 위한 데이터
- Model : Data를 transform한다.
- Loss Function : Model의 badness를 나타낸다.
- Algorithm : loss를 최소화하는 파라미터를 갱신한다.

#### 1-2-1. Data
- 해결하고자 하는 문제에 의존한다.
- 분류, Semantic Segmentation(Pixels), Detection, Pose Estimation, Visual QnA 등이 있다.

#### 1-2-2. Model
- 문재를 해결하기 위한 테크닉적 요소
- AlexNet, ResNet, LSTM, GAN등이 있다.

#### 1-2-3. Loss Function
- Loss Function은 우리가 달성하고자 하는 대략적인 것이다.
- 모델과 데이터가 정해져 있을 경우, 모델을 어떻게 학습해야할지에 대한 것이다.
- Deep Learning Model은 어떤 식으로든 Neural Net의 형태이다.
- Weight와 Bias로 구성되어 있다.
- Weight을 어떻게 업데이트할 지에 대해 기주이 되는 Loss Function을 정의
- 우리의 목적은 학습하지 않은 데이터에 대해서도 잘 작동하게 하는 것. -> 즉, Loss Function이 줄어든다고 해서 우리가 원하는 것을 항상 이룰 수 있는 것은 아니다.
- 우리의 문제에 맞게 어떤 Loss를 사용할지, 왜 사용하는지, 그것이 우리 문제에 어떻게 도움을 주는지에 대하여 알고 있는 것이 중요하다.
- Loss Function에는 MSE, Cross Entropy, MLE 등이 있다.
- Loss Function을 최소화하기 위해 사용하는 방법론이 Optimizer이다.
- Optimizer에는 SGD, Momentum, NAG, Adagrad, Adadelta, Rmsprop 등이 있다.
- Overfitting을 방지하기 위해, Regular를 추가한다.
- Regular에는 Dropout, Early Stopping, K-Fold Validation, Weight Decay, Batch Normalization, MixUp, Ensemble, Bayesian Optimization이 있다.

## 2. Historical Review
- 어떤 딥러닝 방법론들이 큰 임팩트가 있었는지에 대하여 알아본다.
- 어떤 흐름, 지금의 위상이 어떻게 생기게 되었는지에 대하여 알아본다.

### 2-1. AlexNet (2012)
- Convolution Neural Network
- 244 x 244 image 분류
- Image 분류에서 Deep Learning을 활용하여 처음으로 1등
- 역사적인 Black Magic(왜 잘되는 지 모르는데 잘될거야)이 실제적인 성능을 발휘

### 2-2. DQN (2013)
- 알파고, 벽돌꺠기 등에 활용된 강화학습
- Q-Learning 활용

### 2-3. Encoder & Decoder, Adam (2014)
- 문장 번역
- Sequence to Sequence
- 기계어 번역의 트렌드의 변형을 초래
- Hard ware적 문제 -> Adam은 일반적으로 결과를 잘 나오게 하는 Optimizer
- 논문에는 왜 이러한 Learning late를 썻는지, Optimizer를 썻는 지에 대한 설명이 없다. -> HyperParameter Search를 통해 최적의 파라미터를 활용한다.

### 2-4. GAN, ResNet (2015)
- 이미지를 어떻게, 텍스트를 어떻게 만들어 낼 수 있을지에 대한 이야기
- Network가 Generato, Discriminator 두 개를 만들어서 학습한다.
- ResNet : 딥러닝의 딥러닝을 가능하게 만들었다.
- 이전에는 Overfitting의 문제로 Layer를 깊게 쌓는 것이 불가능하였는데, ResNet을 바탕으로 더 깊게 Layer를 쌓아도 test data의 성능을 좋게 만들 수 있다.

### 2-5. Transformer (2017)
- 웬만한 RNN을 대채한다.
- Vision에도 사용할 수 있다.

### 2-6. BERT
- Transformer 활용
- Bidirectional Encoder 활용
- Fine-Tuned NLP Models (대량의 Corpus를 pre-training, 내가 학습하고자 한 단어들으 활용해 Fine Tuning한다.)

### 2-7. Bg Languag (GPT-X) (2019)
- Fine-Tuned NLP MOdel의 끝판왕
- 약간의 Fine Tuning을 활용ㅇ하여 다양한 문장, 프로그램, 표 등을 만든다. (Sequential Model)
- 굉장히 많은 Parameter가 존재한다.

### 2-8. Big Language Models
- SimpleCLR
- Unsupervised Dataset을 함께 활용하여 학습
- 내가 풀고자하는 문제에 대하여 잘 알고, 도메인이 있을 경우 데이터셋을 만들어서 활용한다. (Self Supervised Data Sampling)

## 3. Neural Network
- Neural Networks는 동물의 뇌를 구성하는 생물학적 신경망에 영감을 받은 애매한 시스템이다.
- 우리가 날고 싶다고 해서 항상 새처럼 날 필요는 없다.
- 함수를 모방하는 Function Approximator이다.

### 3-1. Linear Neural Networks - 1차원, 선형(물론, Multi Dimensional 역시 다음과 같은 방법으로 해결할 수 있다.)
![DL Basic 1번](https://user-images.githubusercontent.com/53552847/128705243-865c306d-402b-43c1-9fef-81ed9fd7d71c.PNG)

![DL Basic 2번](https://user-images.githubusercontent.com/53552847/128705248-69e9473a-3379-4d09-8f05-642eb03e492e.PNG)

![DL Basic 3번](https://user-images.githubusercontent.com/53552847/128705250-90fa9d35-dbfa-4b60-b5df-866ec45df22a.PNG)

### 3-2. Beyond Linear Neural Networks -> Nonlinearity
- Linear을 여러 개 쌓아봤자 Linear이다.
- Nonlinearity가 필요
![DL Basic 4번](https://user-images.githubusercontent.com/53552847/128705252-3c766a72-e48d-4fce-b5b9-8878b1b04f50.PNG)
- 위의 식을 여러 번 반복하게 됨으로부터 더 많은 표현력을 가지게 된다.

### 3-3. Activation Functions(Nonlinear Transfrom)
- Rectified Linear Unit(ReLU)
- Sigmoid (0~1 사이 제한)
- Hyperbolic Tangent(-1~1)

### 3-4. Neural Network가 왜 잘되는가?
- 인간의 뇌를 본 떳기 때문에 잘된다!
- Universal Approximation Theorem
    - 이론적으로 Hidden Layer가 1개 있는 Neural Network는 어떤 대부분의 Continuous한 Measurable Function을 근사할 수 있다.(여기서의 근사는 단순한 근사가 아닌 우리가 원하는 어떤 근사까지 이다.)
    - Hidden Layer가 1개만 있는 Neural Network의 표현력은 우리가 일반적으로 생각할 수 있는 대부분의 Continuous한 Function들을 다 포함한다. 그렇기에 잘 작동한다.
    - **Caution :** 이 이론은 단지 존재성만을 보인다. 위에서 설명한 Neural Network가 세상 어딘가에 있다는 것을 말하는 것이지 내가 학습한 Neural Network가 이러한 성질을 가질 것이라고 말해서는 안된다.
    - Neural Network는 다양한 함수를 표현할 수 있을만큼의 표현력을 가지고 있지만 이녀석을 어떻게 찾는지는 모른다.

## 4. 피어 세션

### 4-1. [지난주 질문]

- SVD 특이값 분해

Q. 대각행렬의 갯수를 제한하여 압축하는 방법 : 어떤 기준으로 압축하고 어떻게 적용되는가?
A. (적용흐름은 코드 참조) Sigma-Singular value가 가장 높은 것부터 추출. 즉, 가장 유의미한 정보순으로 나열 후 뒤부터 탈락시키며 압축시킨다.
Q. SVD에서 Simgular value가 원래 크기 순으로 정렬되어 있는가?
A. 더 조사 후 내일 답변.


### 4-2. [이번 주 (금) 발표자 선정]

강진선님, 우원진님

금요일 스페셜 피어세션과 회고로 시간이 부족하다면 다음 피어세션으로 넘김.


### 4-3. [멘토링 시간 변경]

오피스아워 전날 오후 7시(수)로 변경.


### 4-4. [선택과제 1번 스터디]

VIT 실습 코드 스터디. 참조 링크 내부 공유 예정

## 5. 6일차 후기
Basic DL은 기본적으로 DL을 공부했다면 이론적으로는 한 번쯤은 들어봤던 내용을 다시 한 번 설명해주었다. 더불어, 줄이고 줄여서 압축되게 전달을 해주시다보니 분명히 빠진 부분도 있겠지만, 단시간에 많은 내용을 습득할 수 있었다.

실습 및 과제를 진행하면서 이론으로만 들었던 것이 코드로 어떻게 돌아가는지 어느 정도 알 수 있었고 이전에는 Tensorflow만 다뤘는데 Pytorch를 활용해 다룸으로써 보다 폭넓은 이해가 가능했던 것 같다.

마지막으로,, 역시나 선택과제의 난이도는 엄청 높았다.. VIT만 진행하였는데도 이미 정신은 나간지 오래다 ㅜㅜ  그래도 팀원들과 함께 진행함으로서 어느정도 진도를 나갈 수 있었던 것 같다!

이번주도 파이팅!

## 6. 해야할 일
- Further Question
- VIT 논문 리뷰
