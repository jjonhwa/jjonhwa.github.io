---
layout: post
title: "Ustage Day 13"
categories: boostcamp
tags: main
comments: true
---
PyTorch 구조에 대하여 학습한다.

**부스트 캠프 13일차 학습 요약**
- **행사** : 마스터 클래스(최성철 교수님)
- **학습** : 모델 불러오기, Monitoring tools for PyTorch, 과제
- **피어 세션**

## 목차
- [1. 모델 불러오기](#1-모델-불러오기)
- [2. Monitoring tools for PyTorch](#2-monitoring-tools-for-pytorch)
- [3. 피어 세션](#3-피어-세션)
- [4. 13일차 후기](#4-13일차-후기)
- [5. 해야할 일](#5-해야할-일)

## 1. 모델 불러오기
- PyTorch 모델 저장 및 불러오는 방법을 학습한다.
- Transfer Learning을 학습한다.
- state_dict의 의미와 모델 파라미터를 여러 정보들고 함께 저장하고 불러오는 법을 학습한다.
- Pre-Trained Model 활용법을 학습한다.

### 1-1. model.save()
- 학습의 결과를 저장하기 위한 함수
- 모델 형태(Architecture)와 Parameter를 저장
- 모델 학습 중간 과정의 저장을 통해 최선의 결과 모델을 선택
- 만들어진 모델을 외부 연구자와 공유하여 학습 재연성 향상
- model을 저장할 때 사용하는 코드는 다음과 같다.
![pytorch 11](https://user-images.githubusercontent.com/53552847/130157278-c6e31079-4e51-4c31-966c-82b0c4aae989.PNG)
- model.save()를 활용하여 모델을 저장할 때, OrderedDict 형태로 저장이 되며, 이 덕분에 모델을 불러오기가 한 층 쉽다. 
- 모델은 'pt' 확장자로 많이 저장한다. ('pth'도 있지만, 'pt'를 권장)
- load_state_dict를 활용하면 Parameter만 load 가능
- `from torchsummary import summary`를 사용하면 좀 더 깔끔하게 model 정보를 print해서 볼 수 있다.
- state_dict로 저장되어 있을 경우, 동일한 형태의 모델로 선언해주고 여기에 load_state_dict를 해야 Parameter가 제대로 입력된다.

### 1-2. checkpoints
- 학습의 중간 결과를 저장하여 최선의 결과를 선택
- earlystopping 기법 사용 시, 이전 학습의 결과물을 저장
- loss와 metric 값을 지속적으로 확인하고 저장
- 일반적으로 epoch, loss, metric을 함께 저장 (흔히, 저장되는 모델 파일명에 입력하여 기록한다.)
- 특히, Colab에서는 지속적인 학습을 위해 필수적 요소이다.
- checkpoint를 사용하는 코드는 다음과 같다.
![pytorch 12](https://user-images.githubusercontent.com/53552847/130157281-b22a9972-3ca1-4382-809c-dd416b55f0f8.PNG)

**Note :** `BCEWithLogitsLoss()`는 Binary Crossentropy를 의미하며, loss function으로 이를 사용하게 되면, 모델의 마지막 layer에 sigmoid 함수를 추가하여 사용하게 된다.

### 1-3. Transfer Learning
- 다른 Dataset으로 만든 모델을 현재 Data에 적용
- 일반적으로 대용량 데이터셋으로 만들어진 모델의 성능이 좋다.
- 현재의 DL에서는 가장 일반적인 학습 방법.
- Backbone Architecture가 잘 학습된 모델에서 일부분만 변경하여 학습을 수행
- 예를 들어, Image Net에서 학습시킨 모델에 Dataset만 Dogs and Cats로 바꿔 넣어서 마지막 Linear 함수만 수정해주어도 좋은 결과가 나타난다.
- Freezing은 pre-trained model을 활용할 때, 모델의 일부분을 얼리는 기법이며 다음과 같은 양상을 띈다.
![pytorch 13](https://user-images.githubusercontent.com/53552847/130157283-1edbf0ad-c91f-495b-bc6b-425490258e0c.PNG)
- Transfer Learning의 코드는 다음과 같다.
![pytorch 14](https://user-images.githubusercontent.com/53552847/130157284-754b0d64-9a0f-4e0a-bf5e-9be04cac4d1d.PNG)
- 위의 코드에서 볼 수 있듯이, frozen의 경우 model.parameters()로 부터, 각 Parameter들의 requires_grad를 False로 함으로서 진행할 수 있다.
![pytorch 15](https://user-images.githubusercontent.com/53552847/130158313-4173a585-976e-44cc-b0d6-aab07f7a0dcf.PNG)
- 위의 코드에서 볼 수 있듯, 만들어진 모델의 이름으로 접근해서 layer를 수정할 수 있다.
- 하지만, 기존의 모델을 수정하는 것은 지양한다고 한다.

## 2. Monitoring tools for PyTorch
- PyTorch를 이용한 학습 시 metric 등을 기록할 때, 사용할 수 있는 Tensorboard, weight&biases를 학습한다.
- 이러한 Monitoring tool을 통해 DL 모델 학습 실험들을  Parameter와 Metric을 자동으로 저장하는 실험 관리 프로세스를 익힌다.
- 코드 Versioning, 협업 관리, MLOps의 전체적인 흐름을 확인할 수 있다.

### 2-1. Tensorboard
- Tensorflow의 프로젝트로 만들어진 시각화 도구
- 학습 그래프, metric, 학습 결과의 사각화 지원
- PyTorch도 연결 가능 -> DL 시각화 핵심 도구
- 사실상 DL 시각화의 핵심이자 기본도구이다.
- Tensorboard는 다음과 같은 값들을 저장할 수 있다.
    - scalar : metric 등 상수 값의 연속(epoch)을 표시
    - graph : 모델의 computational graph 표시
    - histogram : weight 등 값의 분포를 표현
    - image & text :  예측 값과 실제 값을 비교 표시
    - mesh : 3D 형태의 데이터를 표현하는 도구
- Tensorboard를 보여줄 때 사용하는 코드는 다음과 같다.
![pytorch 16](https://user-images.githubusercontent.com/53552847/130157286-244a949f-028a-48d5-ad03-392f8a3321d8.PNG)
- weight 값을 조정할 때, tensorboard의 histogram을 많이 사용한다.
- tensorboard의 이미지의 경우 그대로 사용하기는 어렵고, 보통은 오답노트와 같은 방식로 왜 틀렸는지에 대하여 적어놓고 이에 따라 학습할 때, 어떤 식으로 모델을 변형시켜줄지, 데이터를 추가해줄지에 대한 전략을 세우게 되는데 이럴 때 주로 사용한다고 한다.

### 2-2. weight & biases (wandb)
- 머신러닝 실험을 원활히 지원하기 위한 상용도구
- 협업, code versioning, 실험 결과 기록 등 제공
- MLOps의 대표적인 툴로 저변 확대 중
- weight & biases를 사용하는 코드의 경우 다음과 같으며, 다음의 코드를 입력하기 전에 project를 사전에 정의해줘야하고 wandb의 API key를 활용해 연결시켜주어야 한다.
![pytorch 17](https://user-images.githubusercontent.com/53552847/130157288-b47d59c2-05d7-4fe0-9bad-78008b2829c3.PNG)

## 3. 피어 세션
### 3-1. 오늘의 질답 및 과제 분석
- (지난 질문) Custom data loader - batch_size 문제
    - Tensor는 concatenate를 통해 value 삽입. ~ 가장 긴 값을 찾고 순차 삽입.
    - functional.pad 사용
    - hstack 사용
    
- 과제
    - 흑마법 구현 : forward.hook 사용
    - TitanicDataset 접근 : 성욱님의 코드 리뷰.
    - __getitem__ 에서 학습 데이터가 아닐 경우, y를 반환하지 말라? : y는 라벨. train=True 여부를 판단하고 getitem에서 조건문을 사용하기.
    - init 함수 구현에 대한 자세한 설명 요약 : 학습 레이블을 잡았을 때, 레이블을 제외한 것들이 features(데이터 목록(string)). ~ 멘토님께 이어 질문하기.

### 3-2. 논문 리뷰
- Transformer (Attention is All needs you)
- 구성 흐름 및 질답(Q, K, V 생성) 진행.

### 3-3. 멘토링 질문
- 다음주면 이미지 분류 대회. 어떤 모델을 사용할지, 참조 자료가 있을지?
- 데이터셋에 관한 질문.

## 4. 13일차 후기
강의가 많이 어렵진 않았고, 유용한 내용들과 코드 실습을 통한 강의여서 이전 강의들에 비해 조금은 편한히 들을 수 있었다. 교수님의 설명력 덕분인지 귀에 쏙쏙 박혔고, 오늘 강의는 나름 재미있었던 것 같다!

강의를 듣는 시간을 제외하고, 밥먹는 시간 빼고는 전부 과제에 몰두 했던 것 같다. 정말이지, 높은 질과 양을 모두 가지고 있는 Quality 있는 과제였던 것 같다. 과제를 통해서 내 스스로가 성장하고 있다는 것을 느낄 수 있었다. 더불어 케바케겠지만 몇 시간이고 붙잡고 있으면 안될거 같은게 어찌어찌 되긴한다라는 것도 깨달은것 같다.. 

피어세션을 통해 팀원들과 과제에 대한 질답도 진행하고 늦은시간까지 연락을 주고받으며 정보들을 조합하여 과제를 잘 해결할 수 있었던 것 같다! 

## 5. 해야할 일
- 과제 정리
- 실습 진행
