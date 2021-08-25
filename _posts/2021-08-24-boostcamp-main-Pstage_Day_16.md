---
layout: post
title: "First P-stage 2(Day 16)"
categories: boostcamp
tags: main
comments: true
---
Data Feeding에 대하여 학습한다.

**부스트 캠프 16일차 학습 요약**
- **학습** : Dataset, Data Generation
- **P-Stage** : transfer learning, EDA
- **피어 세션**

## 목차
- [1. Dataset](#1-dataset)
- [2. Data Generation](#2-data-generation)
- [3. P-Stage](#3-p-stage)
- [4. 피어 세션](#4-피어-세션)
- [5. 16일차 후기](#5-16일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Dataset
- 이미지의 경우 비교적 전처리 할 것이 많지 않다.
- 정형 데이터나 텍스트 데이터의 경우, 상상을 초월하는 전처리가 필요하다.
- 문제 정의가 매우 중요한 요소로 활용된다.

### 1-1. Pre-processing
- Pre-Processing이 80%, Modeling 등이 20%라고 얘기될 정도로 Pre-processing이 많은 비중을 차지하고 중요하다.
- 좋은 데이터를 만들기 위한 과정이며, 이 과정이 가장 중요하다.
- Competition Data는 품질이 매우 양호한 편이다.

### 1-2. 이미지에서의 생각해 볼 수 있는 전처리
- Bounding Box
    - 가끔 필요이상으로 많은 정보를 가지고 있다. 필요없는 부분을 잘라서 사용한다.
    - 즉, Target, Noise로 나타낼 수 있다.
    - Bounding Box가 주어지지 않았을 경우, 이미지의 일관적인 분포를 파악하여 crop을 진행하는 것도 좋은 방법 중 하나이다.
- Resize
    - 계산의 효율을 위해 적당한 크기로 사이즈 변경
    - 이미지의 크기가 너무 크게 되면 계산과정이 복잡해지고 오래걸리게 된다.
- 위의 두 가지 뿐만 아니라 다양한 이미지 전처리 방법이 있다.

### 1-3. Generalization
- Bias & Variance
    - High Bias : Underfitting (Data를 너무 고려하지 않았다.)
    - High Variance : Overfitting (Noise까지 Fitting 해버렸다.)
- Train / Validation
    - 학습 할 때, 제대로 학습됬는지 검증할 수 없기 때문에, 학습에 이용되지 않은 Validation Data를 가지고 어느 정도 일반화 시킬 수 있다.
    - Train Data를 Train, Validation Data로 나누어 학습시켜준다.
- Data Augmentation
    - 주어진 데이터가 가질 수 있는 Case(경우), State(상태)의 다양성이 존재한다.
    - torchvision.transforms
    - Albumentations

## 2. Data Generation
### 2-1. Overview
- 데이터를 잘 구성했다 하더라도, 잘 출력하지 못하면 실속이 없다.

### 2-2. Data Feeding
![1주차 3](https://user-images.githubusercontent.com/53552847/130800828-68319baa-1731-47b9-bf3b-95a5c54afd78.PNG)
- 모델을 학습할 때, 위와 같은 그림을 잘 인지하고 있어야 한다.
- 첫번째의 경우, 아무리 Model이 좋아도, Data Generator가 안 좋다면 빠르게 학습할 수 없다.
- 반대로, 두 번째처럼 Data Generator의 처리가 Model보다 많게 되면 알맞게 활용할 수 있다.
- 이러한 것들은, Batch_size별로 시간이 얼마나 걸리는지를 학인해보고 알맞게 활용할 수 있어야 한다.
- 예를 들어, Resize를 활용해 크기를 키운 후, RandomRotation과 같은 처리를 진행한 것과 RandomRotation을 한 후에 Resize를 활용해 크기를 키우는 것은 많은 성능의 차이가 발생한다.
![1주차 4](https://user-images.githubusercontent.com/53552847/130809176-b3cccf22-44a4-4511-bf8d-c5a329bbe900.jpg)

### 2-3. torch.utils.data
- Dataset
    - 내 Custom Class에 상속 받게 된다.
    - 이를 상속받게 되면, Dataset 행세를 할 수 있다고 이해할 수 있다.
    - `__init__` : Custom Dataset 클래스가 처음 선언 되었을 때 호출
    - `__getitem__` : Custom Dataset의 데이터 중 index 위치 아이템을 리턴
    - `__len__` : CUstom Dataset 아이템의 전체 길이
    - 위의 3개가 Dataset에서 가장 기본이 되는 함수이다.
- DataLoader
    - 내가 만든 Dataset을 효율적으로 사용할 수 있도록 관련 기능을 추가하는 것이다.
    - batch_size, num_workers, drop_list,sampler, collate_fn, shuffle 등등이 있다.
    - 이 중에서, batch_size, smapler, collate_fn, shuffle은 자주 사용한다.
    - collate_fn
        - batch 마다 함수를 다르게 주고 싶을 경우 사용한다.
        - 취합하는 함수를 넣고 싶을 때, 사용한다.
        - 연구적으로 사용하거나, 데이터가 복잡해지는 경우 많이 사용한다.
- Dataset과 DataLoader는 엄연히 다른 일을 하며, 분리하여 사용하는 것이 좋다.
- DataLoader는 하나만 만들어놓고 Dataset만 바꿔서 재사용할 수 있다.

## 3. P-Stage
- Transfer Learning
    - Resnet18, vgg16을 활용하여 학습
    - classification 부분만 `requires_grad = True`로 하여 학습
-> 학습 코드를 엉망으로 짜서 확률이 0.0003과 같이 나왔다. 수정 필요....
- EDA
    - Special Mission으로 오픈된 EDA를 따라서 코딩 진행하였다.
    - 전처리를 하기 위한 다양한 아이디어를 제공해주었으며 추후에 적용 및 적용방법을 연구해볼 필요성이 있다.
- 데이터 노이즈 확인
    - AI Stages의 토론 게시판을 통해 기존의 데이터에 노이즈가 존재함을 알 수 있었다.
    - 이를 통해, 기존의 데이터를 수정해줄 필요성을 느낄 수 있었고, 이를 처리한 후에 다시 전처리 및 모델링을 진행해 볼 것이다.

## 4. 피어 세션
### 4-1. [마스크 데이터 분류 대회]
- 어느 부분에 집중하고 있는지?
    - 모델
    - 전처리
- batsize에 대한 custom data 처리 방식
    - collate_fn 사용해줬다. 질문: collate_fn은 필수인가? <- DataLoader를 사용하면 colalte_fn은 필수가 아니다.
    - x와 label를 따로 따로 해주었다.
- Label 새로 생성
    - glob 사용
    - 규칙을 발견해서 코드 작성
- DataLoader
    - 이미지 폴더 모듈 관련 :torchvision안에 있는 ImageForder 
        - 장점 : labeling이 어느정도 되어 있어서 전처리시 편리
        - 단점 : 복사로 인한 메모리 차지 
    - tranforms 함수 중에서-  getitem관련 문제 
- 모델 관련
    - VGG  + 2 layer추가 = 높은 validation 값
    - wild-resnet : train set 에서 90%까지 나옴 but overfiting 조심해야 된다.
    - 정확도가 터무니없게 나올때 어디부분이 잘못된 것인지 확인할 수 있는 방법?
- 학습 관련
    - train, validation split 
    - transform에서 normalization해주는 이유
        - 불균형을 해결할 수 있다.
        - 속도가 빨라진다. 
        - 예를 들어서 MNIST에서 반전시킬 경우255값에 치우쳐있게 된다. 이럴때 분산에 정도를 줄여줄 수 있다.
- EDA 활용에 대해서
    - 마스크별 RGB histogram,  마스크 착용여부 RGM histogram 
    - open cv의 casecadeclassifcation 이용 -> BUT 예전에 만들어진 거라서 성능이 좋지 않다.
    - open cv의 detect_face, detect_gender 이용하면 좋다. <- 모든 이미지에 대해서 bounding box를 찾아서 활용하면 좋지 않을까 싶다. 잘못된 라벨링에 대한 문제를 해결할 수 있다.
    - PCA를 활용해보자
    - 공개 코드 -> range(len())부분 오타 : 지워주면 될 것 같다.
- 기타
    - CrossEntropy함수 안에서 자동으로 softmax를 취해준다.

## 5. 16일차 후기
P-Stage 2일차... 멘탈이 탈탈 털린지 오래이다.. 아무리 코드를 수정해도 정확도는 0을 맴돌고 무엇이 잘못된지 제대로 짠건 만는지도 알 수 없었다. 피어세션을 통해 팀원들에게 질문공격을 해서 어느 정도 도움을 받을 수 있었지만 아직도 코드가 제대로 돌아가고 있지는 않다. 문제점을 하루 빨리 찾아내서 수정을 해야겠다..

## 6. 해야할 일
- 이미지에서의 생각해 볼 수 있는 전처리에 대하여 생각해보기
- DataLoader의 기능 찾아보기
