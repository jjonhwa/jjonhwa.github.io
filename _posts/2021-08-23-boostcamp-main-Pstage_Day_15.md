---
layout: post
title: "First P-stage 1(Day 15)"
categories: boostcamp
tags: main
comments: true
---
Competition에 대한 소개, 각 구성요소에 대하여 학습한다.

**부스트 캠프 15일차 학습 요약**
- **행사** : 타운홀 미팅
- **학습** : Competition with AI Stages, Image Classification & EDA
- **P-Stage** : torch.utils.data.Dataset, torch.utils.data.DataLoader, nn.Module
- **피어 세션**

## 목차
- [1. Competition with AI Stage](#1-competition-with-ai-stage)
- [2. EDA](#2-eda)
- [3. P-Stage](#3-pstage)
- [4. 피어 세션](#4-피어-세션)
- [5. 15일차 후기](#5-15일차-후기)

## 1. Competition with AI Stage
### 1-1. Overview 확인하기
- 내가 지금 풀어야할 문제가 무엇인지 확인하기
- 해결하고자 하는 문제의 Input과 Output이 무엇인지 확인하기
- Solution은 어디서 어떻게 사용되는가
- Data Description을 읽기
- Discussion을 무조건 활발히 잘 이용하자!!

### 1-2. 문제 해결과정
![1주차 1](https://user-images.githubusercontent.com/53552847/130800816-5dfd73ad-21c5-48f4-8f64-cbee0356525d.jpg)
- 전반적인 문제 해결과정은 위의 그림과 같으며 이 중 Competition에서 겪게될 과정은 다음과 같다.
    - Domain Understanding(Overview)
    - Data Analysis
    - Data Processing
    - Modeling
    - Training

## 2. EDA
### 2-1. EDA란?
- Exploratory Data Analysis
- 데이터를 이해하기 위한 노력
- 답은 정해져 있지 않으며, 끊임없는 물음을 해결하기 위한 과정이다.
![1주차 2](https://user-images.githubusercontent.com/53552847/130800824-22f0493a-1f16-42ca-ad5d-23e6427a3390.jpg)
- 위의 그림처럼, Domain Understanding 이후에, 처음 진행해야하는 과정이다.
- 자신의 생각을 채워넣는 서술형 문제라고 생각하자.

## 3. P-Stage
- torch.utils.data.Dataset
    - Dataset을 활용하여 Data를 Generator형태로 만들어주었다.
    - `__init__`, `__len__`, `__getitem__`을 활용하여 처음부터 코딩하여 만들어주었다.
- torch.utils.data.DataLoader
    - Dataset을 모델에 Feeding 시켜주기 위해 DataLoader를 만들어주었다.
    - shuffle, batch_size 등의 option을 주어 만들어 주었다.
- nn.Module
    - Submission으로 미리 주어진 모델 코드를 확인하고 이해하였다.

## 4. 피어 세션
### 4-1. [마스크 데이터 분류 대회]
- 마스크 데이터 분류 대회에 대하여 토의함

### 4-2. [데이터 전처리]
- center crop, background masking, 등의 방법이 제안됨.
 
### 4-3. [DataLoader]
- 확장자 뿐 아니라 속성도 다양함. → glob 이용 시 편리함.
- 데이터를 읽는 방법에 관하여 토의함: map-style, iterable-style
 
### 4-4. [EDA]
- 각 속성(성별, 나이) 에 관련한 데이터 분포를 시각화 함.

### 4-5. [모델 관련]
- 주어진 input으로부터 label 유추를 어떻게 해내는지 궁금하다.
- 레이블링의 방법: classification / regression
→ 주어진 age 속성을 정형 데이터로 이용하는 방법에 관하여

- 따라서 피어세션 진행 형식을 미리 정하면 계획적으로 시간을 활용할 수 있을 것
- end-to-end 학습 방법을 이용해도 되는지 토의해 봄.
→ 마스크 착용 여부, 나이, 성별의 독립적인 속성이 섞여 있기 때문.

### 4-6. [학습정리]
- 배운 것, 시도한 것 등을 기입할 것.

## 5. 15일차 후기
P-Stage 첫 날이라 그런지, 강의는 되게 교양수업같은 느낌이었다! 처음이라 Overview 및 Competition에 대한 내용을 듣다 보니 가볍게 들을 수 있던 것 같다.

드디어 P-Stage가 시작했고 코딩을 직접적으로 들어간다는 생각에 조금은 설레였던 것 같다!! 

정말, 아주 조금이었던 것 같다. 이런 생각이 들었다는 것이 무색할 정도로, 아무런 Baseline 없이 코드를 짠다는 게 이렇게 어려운 건지 처음으로 알 수 있었다.... Dataset을 불러오는 것부터, Model에는 어떻게 입력해줘야하는지,, 코딩초보인 나로서는 도통 알 수 없었고 지난 과제들의 필요성, 중요성이 머리를 쓱 훑고 같다..

길지 않은 강의에 조금은 여유롭겠다고 생각했는데, 여유는 무슨.. 그냥 계속 코딩했던 것 같다. 어떤 결과도 나오지 않았는데 구글링과 지난 과제들을 뒤져보며 어떻게든 데이터를 불러오고 정제하겠다는 마음하나로 임했던 것 같다.

정말 멘탈이 탈탈 털렸고 그나마 위안 삼을 수 있는 것은 성공적으로 Dataset과 Dataloader를 만들 수 있었던 것이다.. 모델은 언제 만들고, HyperParameter Tuning? 할 수 있는거 맞겠지... 데이터 전처리는 또 언제 ㅜㅜ
