---
layout: post
title: "Ustage Day 11"
categories: boostcamp
tags: main
comments: true
---
Generative Model에 대한 설명을 진행한다.

**부스트 캠프 10일차 학습 요약**
- **행사** : 특별강의(개발자로 산다는 것)
- **학습** : Introduction to PyTorch, PyTorch Basics, PyTorch 프로젝트 구조 이해하기
- **피어 세션**

## 목차
- [1. Introduction to PyTorch](#1-introduction-to-pytorch)
- [2. PyTorch Basics](#2-pytorch-basics)
- [3. PyTorch-프로젝트-구조-이해하기](#3-pytorch-프로젝트-구조-이해하기)
- [4. 피어 세션](#4-피어-세션)
- [5. 11일차 후기](#5-11일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Introduction to PyTorch
- 딥러닝 프레임워크의 종류, PyTorch의 기본적인 작동 구조에 대하여 학습한다.
- Framework를 공부하는 것이 사실상 DeepLearning을 공부하는 것이다.

### 1-1. Keras, Tensorflow and PyTorch
![pytorch 1](https://user-images.githubusercontent.com/53552847/129735667-aa0241fe-cd35-42a0-86bf-06235e6f1613.PNG)

#### Keras & Tensorflow
- Keras는 Wrapper 언어이다.
- Keras는 Row-Level 언어의 사용을 좀 더 용이하게 만들어준다.
- Keras와 Tensorflow는 사실상 합쳐져 있다고 해도 무방하다.
- Keras가 Tensorflow 2.0에서는 그대로 흡수되어 있다.

#### PyTorch
- Row, High Level을 둘 다 지원하는 언어
- BackPropagation을 할 때, 미분을 진행해야 하며, 이를 그래프로 표현할 수 있어야 하는데, 이 때, PyTorch의 경우 Dynamic Computation Graph를 지원하며 이것이 Tensorflow와의 가장 큰 차이점이다.

### 1-2. Compuational Graph
- 연산과정은 다음과 같다.
![pytorch 2](https://user-images.githubusercontent.com/53552847/129735673-a148dfa6-077b-4a23-956a-eb279861dd78.PNG)

#### Define and Run (Tensorflow)
- 그래프를 먼저 정의한다. (그래프를 그리는 코드를 먼저 작성)
- 실행 시점의 데이터를 그래프에 Feed 시켜준다.

#### Define by Run (PyTorch, DCG)
- 실행을 하면서 그래프를 그린다.
- 일반적으로 속도가 느릴 것 같지만, 대부분의 경우 PyTorch가 체감상 Tensorflow보다 빠르다.
- Debug에서 큰 장점을 가진다. 

### 1-3. Tensorflow vs PyTorch
|Tensorflow|PyTorch|
|---|---|
|Production, Cloud 연결, Multi-GPU, Scalability|개발 과정에서의 쉬운 Debugging, 논문 작성 및 아이디어 구현 용이, Define by Run (Pythonic Code), GPU Support, Good API and Community, 사용하기 쉽다.|

### 1-4. PyTorch의 특징
- Numpy + AutoGrad + Function
- Numpy 구조를 가지는 Tensor 객체로 Array 표현
- 자동 미분을 지원하여 DL 연산을 지원한다.
- 다양한 형태의 DL을 지원하는 함수와 모델 지원 (Dataset, Multi-GPU, Data-Augmentation etc)

## 2. PyTorch Basics
- PyTorch의 기초 문법을 배우게 된다.
- Tensor 사용법 및 자동 미분기능인 AutoGrad에 대해 학습한다.
- Custom Network를 작성할 수 있는 기본 지식을 익힌다.

### 2-1. Tensor
- 다차원 Array를 표현하는 PyTorch 클래스
- 사실상 Numpy의 ndarray -> PyTorch의 Tensor
- Tensor를 생성하는 함수도 거의 동일하다.(다음과 같다.)
![pytorch 3](https://user-images.githubusercontent.com/53552847/129735675-56a5955b-d52b-47d6-a946-d8df3d11a99d.PNG)
- `t_array = torch.FloatTensor(n_array)`, `tensor_array = torch.from_numpy(nd_array_ex)`, `x_data = torch.tensor(data)`와 같이 tensor를 만들 수 있으나, 직접 생성해서 사용하는 경우는 많이 없다.

### 2-2. Tensor Data Type
- 사실상 Numpy와 동일
- 차이는 GPU Tensor이다.
![pytorch 4](https://user-images.githubusercontent.com/53552847/129737790-d6acac11-b97a-443d-8ac5-8f5bce6d38c6.png)

### 2-3. Tensor Handling
- view : reshape과 동일하게 tensor의 shape을 변환
    - reshape과 거의 동일
    - 메모리 형태를 보장해준다. 그래서 다른 입력값에 넣어줬을 경우, 메모리 참조 형태이므로 함께 변하게 된다.
- squeeze : 차원의 개수가 1인 차원을 삭제(압축)
- unsqueeze : 차원의 개수가 1인 차원을 추가
![pytorch 5](https://user-images.githubusercontent.com/53552847/129738682-3e61e28b-2a78-40a1-9e9d-e15c7d8fc844.png)

### 2-4. Tensor Operation
- 내적을 구할 때, dot이 아닌 mm을 사용한다.
- 연산 되는 값이 모두 벡터 혹은 scalar일 경우 dot을 사용하며, 행렬일 경우에는 mm을 사용한다.
- mm은 Broadcasting을 지원하지 않는다.
- matmul은 Broadcasting을 지원한다.
![pytorch 6](https://user-images.githubusercontent.com/53552847/129738689-c68259ec-9e2e-4118-a559-d99856c99647.png)

### 2-5. Tensor Operations for ML/DL Formula
- nn.functional 모듈을 통해 다양한 수식 변환을 지원한다.
- 다 외울 필요는 없으며, 찾아보면서 활용할 수 있도록 익히자.

### 2-6. AutoGrad
- PyTorch의 핵심이며 backward함수를 사용한다.
- tensor에서 미분이 되는 대상은 requires_grad = True로 준다.

## 3. PyTorch 프로젝트 구조 이해하기
- PyTorch 학습 프로젝트 구성에 대해 학습한다.
- OOP와 모듈의 개념을 통해 프로젝트를 구성하는 방법을 살펴본다.
- VSCode와 SSH로 연결해서 Colab을 사용하는 방법 등 딥러닝 학습을 위해 환경을 구축하는 방법도 다룬다.

### 3-1. PyTorch Project Template Overview
- 초기 단계에서는 대화식 개발 과정이 유리하다.
    - 학습과정과 디버깅 등 지속적인 확인이 가능하다.
- 배포 및 공유 단계에서는 Notebook 공유의 어려움이 있다.
    - 재현하는 것이 어렵고, 실행 순서가 꼬인다는 단점이 있다.
- DL 코드도 하나의 프로그램이다.
    - 개발 용이성 확보와 유지보수의 향상이 필요하다.
- 이하 실습.

## 4. 피어 세션
### 4-1. [금일 질문 목록]:

- torch.gather에서의 dim의 의미는?
    - input과 index의 dimension을 동일해야 한다. 만약, input이 4x10x15이고, dim = 0이면, index는 Nx10x15여야 한다. 즉, dim으로 입력한 차원을 제외한 나머지 차원은 동일해야한다.
- chunk의 2차원 배열에서의 변환은 어떻게 이루어지는가?
- nn.Linear에서의 크기 변환을 어떻게 해야하는가?
    - 과제에서 nn.Linear(2,5)를 입력해주게되면, 열의 개수가 2개에서 5개로 변한다.
    - 이렇게 되는 원리에 대해서는 정확히 알지 못했다.
- 3차원에서 대각행렬이면 정육면체에서 전체에 대한 대각인줄 알았는데 아닌가?
    - 본 과제에서 3차원에 대한 대각행렬을 각 2차원을 뱃치로 보고, 각 2차원 행렬에 대한 대각원소를 의미한 것이다.
- expand는 어떻게 활용하는 것인가?
    - 여러 배치가 있다고 했을 때, 각 배치마다 동일한 값을 반복하여 나타내어준다.


### 4-2. [주별 발표]
- 다양한 이미지 Dataset별로 분류 모델을 직접 돌려본 것에 대한 설명을 진행
- Git-Flow에 대하여 다시 한 번 상기시켜주었으며, git 사용법에 대한 발표 진행.

## 5. 11일차 후기
PyTorch에 대한 내용을 학습함으로서, 지난 주에 학습했던 모델들에 대한 코드 이해에 조금은 한발짝 다가갈 수 있었던 것 같다. 핵심만 콕콕 집어주시는 교수님의 강의 덕분이었는지 PyTorch와 한 걸음 가까워진 느낌이 들었다.

과제의 Quality가 갑자기 훅 올라간 느낌이다! 마치 위키독스 같은 느낌이 묻어날정도로 정성스러운 과제라고 느껴졌고 그만큼 어렵기도, 내용이 많기도 했다. 그래도 그만한 output을 가져갈 수 있을 정도의 Quality였다! 

이렇다 할지라도 나는 여전히 전부 해결하지 못했다 ㅜㅜ. 내 실력의 Quality는 아직 못미친다는 것 역시 매일매일 뼈저리게 느끼고 있다...

마지막으로 금일 '개발자로 산다는 것'이라는 주제로 특강이 있었는데 실무에 대한 깊이있는 말씀들을 많이 들을 수 있었고, 개발자로서 내가 어떻게 준비해야하고 어떤 걸 준비해야하는지도 느낄 수 있는 시간이었다.

매주 유익한 시간, 유익한 강의들로 채워나가면서, 내 실력의 밑천, 방대한 양의 공부량을 매번 느끼고 있다...하하. 그래도 꾸준히 해보자! 이제 2주 공부했으니 앞으로 나아갈 시간이야 얼마든지 있다!

## 6. 해야할 일
- Q.backward(gradient = external_grad) -> external_grad = torch.tensor([1., 1.]) : 이것은 무엇을 의미하는 가?

