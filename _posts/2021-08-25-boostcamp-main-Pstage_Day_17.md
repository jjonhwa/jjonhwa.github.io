---
layout: post
title: "First P-stage 3(Day 17)"
categories: boostcamp
tags: main
comments: true
---
Model에 대하여 학습한다.

**부스트 캠프 17일차 학습 요약**
- **학습** : Model with PyTorch, Pretrained Model
- **P-Stage** : Transfer Learning, Layer 추가, KFold적용, Early Stopping 적용
- **피어 세션**

## 목차
- [1. Model with PyTorch](#1-model-with-pytorch)
- [2. Pretrained-Model](#2-pretrained-model)
- [3. P-Stage](#3-p-stage)
- [4. 피어 세션](#4-피어-세션)
- [5. 17일차 후기](#5-17일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Model with PyTorch
### 1-1. 모델이란?
- 일반적으로 Object, Person 또는 System을 대변하는 정보이다.

### 1-2. Design Model with PyTorch
- PyTorch의 가장 큰 장점은 자유롭고, Pythonic하며 연구하기 좋다는 것이다.
- 유연하다.
- From Research to Production
- Low-level, Pythonic, Flexibility
- 코드를 이해함으로서 Training이 어떻게 이루어지는지 어느 정도 알 수 있다.

### 1-3. nn.Module & modules
- PyTorch 모델의 모든 레이어는 nn.Module 클래스를 따른다.
- model.modules()를 사용하면 model에 속해있는 모든 모듈들이 출력된다.
- 이를 활용하여, 모델을 저장하고 불러오는 것까지 할 수 있다.
- 파라미터를 저장하는 곳이다.

### 1-4. nn.Module Family & forward
- nn.Module은 child modules를 가질 수 있따.
- 모든 nn.Module은 forward() 함수를 가진다.
- 내가 정의한 모델의 forward()를 한 번 실행한 것만으로 모델의 forward에 정의된 각각의 forward()가 실행된다.

### 1-5. Parameters
- 모델에 정의되어 있는 modules가 가지고 있는 계산에 사용되는 Parameter
- model.state_dict(), model.parameters()를 활용하여 Parameter의 각각의 tensor를 볼 수 있다.
- 각 모델 파라미터들은 data, grad, requires_grad 등의 변수들을 가지고 있으며, 이를 학습하는데 유용하게 사용할 수 있다.

### 1-6. PyTorch의 Pythonic
- 예를 들어, model.state_dict()는 Python의 Dictionary형태로 구성되어 있기 때문에, Dict의 구조만 파악하고 있다면 얼마든지 응용할 수 있고, 에러에 대한 핸들링도 보다 손쉽게 할 수 있다.

## 2. Pretrained Model
### 2-1. ImageNet
- 획기적인 알고리즘 개발과 검증을 위해 높은 품질의 데이터셋이 필수인데 이를 어느정도 해결해주었다.
- 대용량의 이미지 데이터 셋이다.

### 2-2. Pretrained Model
- 모델 일반화를 위해 매번 수 많은 이미지를 학습시키는 것은 까다롭고 비효율적이다.
- 좋은 품질, 대용량의 데이터로 미리 학습된 모델을 내 목적에 맞게 다음어서 사용하도록 한다.
![1주차 5](https://user-images.githubusercontent.com/53552847/130880392-c86fb7b3-4dc0-44dd-9fb7-6db7cda993c8.PNG)

### 2-3. Transfer Learning
- 내 데이터, 모델과의 유사성을 따져서 적용할 수 있다.
- Pretraining 할 때, 설정했던 문제와 현재 문제와의 유사성을 고려해야한다.
- 대부분의 Pre-Trained Model은 ImageNet을 바탕으로 학습되었으므로, classification도 우리의 분류와 맞게 바꿔줘야하며 ImageNet과의 방향성도 고려해야한다.

### 2-4. Case by Case
- 문제를 해결하기 위한 학습 데이터가 충분하다.
    - High Similarity : CNN Backbone freeze -> Classifier만 수정 (feature extention)
    - Low Similarity : CNN Backbone 수정 -> Classifier 수정 (fine-tuning)
- 학습 데이터가 충분하지 않을 경우.
    - High Similarity : CNN Backbone freeze -> Classifier만 수정
    - Low Similarity : 추천x -> Overfitting, Underfitting의 문제가 생길 확률이 높다.

## 3. P-Stage
- ResNet18을 활용한 Fine-Tuning 진행
- 각 Layer별로 freeze한 후 학습 진행
- F1-Score를 출력으로 나타날 수 있도록 코드 수정
- Val-Loss 기준으로 Early Stopping 적용
- Stratified KFold 적용
    - 현재 주어진 데이터는 Imbalanced Dataset이다.
    - 이 때문에, Stratified KFold를 적용하여 매 Fold마다 모든 Class가 들어갈 수 있도록 설정

## 4. 피어 세션
### 4-1. [마스크 데이터 분류 대회]
- out of memory 문제 -> batch size를 조정하니까 해결이 되었다.
- Hidden file 은 삭제해도 되는가 ? -> 보안때문에 해놓은것으로 올라왔다.
- Training Acc와 Val Acc는 잘 나오는데 Test에서는 잘 안나온다 ? -> 오버 피팅 가능성, 우리 Competition에 Transfer Learning의 사용이 맞는지 잘 모르겠다.
- 데이터 불균형 해소 방법으로 적은 데이터의 갯수에 맞춰서 학습을 시키면 되지않나? -> 이걸 under sampling이라고 하는데 이것은 데이터의 손실이 많아서 이보다는 Over Sampling이나 다른 기법들을 사용하는것이 좋을것 같다.
- Pretrained 모델의 Back bone에서 앞부분의 freeze를 풀고 학습 하면 안되나? -> 원래는 층이 깊어질수록 고차원의 피쳐를 인식할수 있는데 Wide resnet의 경우 깊이를 반을 줄여서, 깊어질수록 좋지는 않다.
- 데이터에서 label이 잘못 된게 있는데 이걸 확인해보고 있다. -> 아예 다 제외하고 학습 할 것인가, 아니면 고칠것인가 ??
- EDA Example을 사용해 전처리를 해보았다. -> CV 라이브러리로 얼굴을 찾아주는 기능을 사용해서 바운딩박스를 찾아보도록 해봤다. -> 이를 통해 데이터 대부분의 얼굴들이 가운데 있었던것 같다. -> 마스크를 쓰고있는 데이터의 경우 이 라이브러리에는 학습이 되어있지 않아서 잘 찾지 못했다.
- 이미지를 흑백으로 바꿔서 학습을 하면 잘 안되는가?? -> 아마 배경의 노이즈에 영향을 많이 받을것 같다.
- 데이터 셋의 Transform은 매 epoch마다 적용된다.

## 5. 17일차 후기
P-Stage 3일차... 드디어 Accuracy가 잘못나오는 문제를 해결했다. 컴퓨터는 거짓말 하지 않는다. 코드를 떼어내서 이해했어야하는데 그대로 옮겨적어 사용하려다 보니 잘 안됬던 것이다. 라인 오류도 있었고, 코드 상의 오류도 있었다. 이들을 해결하니 다른 피어분들 처럼 정확도가 어느 정도 나왔다!

전처리를 하면서도, 모델 구성도 못하는데 전처리하면 뭐하나라는 생각에 멘탈이 탈탈 털렸었는데 모델 구성이 어느 정도 되었으니 전처리 및 모델 수정을 해보면 좋을 것 같다!

한 줄기 희망이 생겼고, 이를 바탕으로 좀 더 열심히 할 수 있는 날이었던 것 같다. F1-Score 적용, Stratified-KFold, Early Stopping을 모델에 적용하여 성공적으로 Training시켰으며 성능도 이전에 비해 아주 조금은 향상된 것 같다. 이제는 전처리 및 다른 Pre-Trained Model에 대한 실험을 진행할 예정이다!

아직 성능이 좋지는 않지만 희망을 가지고 열심히 해보도록 하자!

## 6. 해야할 일
- Imbalanced 문제 해결방법 연구
