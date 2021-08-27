---
layout: post
title: "First P-stage 4(Day 18)"
categories: boostcamp
tags: main
comments: true
---
Training & Inference에 대하여 학습한다.

**부스트 캠프 18일차 학습 요약**
- **행사** : 마스터 클래스
- **학습** : 학습 요소, Training & Inference
- **P-Stage** : ResNet18, ResNet50, ResNext50_32x4d, Augmentation 적용
- **피어 세션**

## 목차
- [1. 학습 요소](#1-학습-요소)
- [2. Training & Inference](#2-training--inference)
- [3. P-Stage](#3-p-stage)
- [4. 피어 세션](#4-피어-세션)
- [5. 17일차 후기](#5-17일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. 학습 요소
### 1-1. Loss
![1주차 6](https://user-images.githubusercontent.com/53552847/131045860-4c497c75-7cc7-42fc-bb6a-4d95148613e8.PNG)
- Loss는 위의 그림과 같은 방식으로 Error를 발생해 Backpropagation에 영향을 미친다.
- Loss는 nn.Module의 상속을 받고 nn.Module의 Family로서의 역할을 한다.
- `loss.backward()`를 사용하여 backward과정을 진행할 수 있으며, 이는 모델 Parameter의 grad값을 업데이트 시킨다.
- Loss란 Error를 만들어내는 과정에 양념을 치는 것이라고 이해하면 조금 쉽다.
- 조금 특별한 몇 가지 loss의 예시를 살펴보자.
    - Focal Loss : Class Imbalance 문제가 있는 경우, 맞춘 확률이 높은 class의 경우 조금의 loss를, 맞춘 확률이 낮은 class의 경우 loss를 높게 부여하는 방식이다.
    - Label Smoothing Loss : class target label을 one-hot 방식(예를 들어, 이진분류 일 때 1 or 0으로 표시)으로 표현하기 보다는 조금 Soft하게 확률로 표현해서 일반화 성능을 높이기 위해 사용한다. 즉, 동시에 다른 피쳐들에 대해서도 점수를 준다고 이해할 수 있다.

### 1-2. Optimizer
- 어느 방향으로, 얼마나 움직일 것인지를 정하며, 영리하게 움직일수록 빨리 수렴한다.
- 다양한 Optimizer에 대한 설명은 [여기](https://jjonhwa.github.io/boostcamp/2021/08/10/boostcamp-main-Ustage_Day_7/#2-practical-gradient-descent-method)를 참고하도록 하자.
- Learning Rate를 동적으로 조절하기 위해 다양한 LR Scheduler가 등장했고 다음과 같은 LR Scheduler들이 있다.
    - Step LR : 특정 Step마다 LR 감소
![1주차 7](https://user-images.githubusercontent.com/53552847/131045848-7086a429-780f-4154-8ca1-01bec84c2ea7.PNG)

    - CosineAnnealingLR : Cosin 함수의 형태와 같이 LR을 급격하게 계속 변경해준다. LR을 급격하게 변화해줌으로서 Local Minimum에서 효과적으로 탈출할 수 있다.
![1주차 8](https://user-images.githubusercontent.com/53552847/131045851-6bc70101-4c59-4b6f-932e-2d9598445d13.PNG)

    - ReducedLROnPlateau : 일반적으로 가장 많이 사용하는 LR Scheduler이고, 주어진 LR로 더 이상의 성능 향상이 없을 때 LR을 감소시키며 진행한다.
![1주차 9](https://user-images.githubusercontent.com/53552847/131045853-ca7aa41f-4123-436a-a915-20ce7e631ee3.PNG)

    - 보통, 변화를 많이 꾀하는 경우에는 Cosine Annealing을 많이 사용하며, 안정적인 학습을 위해서는 ReducedLROnPlateau를 많이 사용한다.

### 1-3. Metric
- 학습에 직접적인 영향을 미치지는 않는다.
- 실제 Production에 쓸 수 있을 정도가 되는지 확인하기 위해서는 Metric이 필수적이다.
- Competition의 경우, Metric을 미리 정해주지만 현업에서는 어느 누구도 정해주지 않으므로, 많은 Competition 경험을 바탕으로 현업에서 적재적소에 알맞은 Metric을 활용할 수 있어야 한다.
- 데이터의 상태에 따라 적절한 Metric을 선택할 수 있어야 한다.
- 다양한 Metric
    - Classification : Accuracy, F1-Score, Precision, Recall, ROC&AUC
    - Regression : MAE, MSE
    - Ranking(특히, 추천시스템) : MRR, NDCG, MAP

## 2. Training & Inference
### 2-1. Training Process
- Traning 준비
    - Training 이전의 과정으로, Dataset -> DataLoader -> Pre-Trained Model -> Loss/Optmizer/Metric을 지정했으며 이제야 Training을 진행할 수 있다.
- model.train()
    - 이 코드로 부터, 모델을 trainable하게 만들 수 있다.
    - train, validation에 대하여 Dropout, Batchnolaization 등이 조금 다르게 동작되어야 하므로, train mode인지 evaluation mode인지로 나뉘게 된다.
    - model.eval()의 경우 validation에 대하여 정의하는데, model.train(False)와 동일하다고 되어있다.
- optimizer.zero_grad()
    - grad값을 초기화 해준다.
    - grad값을 초기화해주지 않으면, 이전 Batch에서 사용하던 grad가 그대로 넘어와 loss.backward()가 진행되므로 일반적으로 초기화 해준다.
- loss
    - loss는 다음과 같이 정의해준다. 
![1주차 10](https://user-images.githubusercontent.com/53552847/131045855-4c747e7f-7adb-4ca2-9527-5b998cda25b3.PNG)
    - loss의 경우(위의 그림에서는 `torch.nn.CrossEntropyLoss()`) 정의하게 되면, 각 gradient들의 chain이 발생하게 되며 `loss.backward()`를 통해 연결된 모든 Parmaeter들의 grad값을 줄 수 있다. 
    - `loss.backward()`를 통해 모든 Parameter들의 grad가 정해지고, `optimizer.step()`를 통해 Parameter값들의 업데이트가 진행된다.
    - 즉, loss는 grad를 업데이트 하고, optimizer는 실제 사용할 Parameter를 업데이트한다고 이해하면 쉽다.
- Training Process의 응용
    - 다양한 응용이 있을 수 있지만 아래의 Gradient Accumulation을 확인해보도록 하자. 
![1주차 11](https://user-images.githubusercontent.com/53552847/131045856-11882dc5-2f3f-4940-9f5f-36abf6a32277.PNG)
    - Gradient Accumulation은 Batch_size를 늘리고 싶은데, 하드웨어의 성능의 제한이 있을 경우 많이 사용한다.
    - 원하는 Batch만큼의 loss를 쌓아주고, optimizer.step()을 원하는 Batch만큼 진행했을 때 진행함으로서 Batch_size를 늘려주는 효과를 가질 수 있다.

### 2-2. Inference Process
- Inference 프로세스의 이해
    - model.eval() : Evaluation mode이며, 앞서 말한 것과 같이 model.train(False)와 동일하다.
    - with torch.no_grad() : torch.set_grad_enabled(False)를 의미하며, 모든 grad_enabled를 False로 만들어준다.
- Validation 셋을 만들어줌으로서 검정을 진행할 수 있다.
- Checkpoint는 다음과 같은 형태로 직접 코딩하여 만들 수 있다.
![1주차 12](https://user-images.githubusercontent.com/53552847/131045857-7b357fc7-8856-4acc-a2bb-9c84ef6f9d3c.PNG)
- 마지막으로, 최종 output, submission형태로 변환하여 만들어 Competition에 제출할 수 있도록 만들어준다.

### 2-3. PyTorch Lightning
- 실제로 현업에서는 생산성이 정말 중요하며, 반복적인 업무를 보다 효율적으로 하기 위해서 등장하였다.
- 위에서 했던 프로세스를 간략하게 Class로 정의해서 불러와 마치 Keras와 같은 형태로 사용할 수 있다,
![1주차 13](https://user-images.githubusercontent.com/53552847/131045859-a56022dd-38b8-479e-a6df-09ae4e7be750.PNG)
- 다만, Deep Learning Process에 대한 이해를 위해서는 위에서 처럼, PyTorch를 활용하여 구현해보는 것이 좋으며, 충분한 이해가 됬을 때에 PyTorch Lightning을 사용하도록 하자.
- 빠르게 개발해야할 경우, 사용하면 좋다.

## 3. P-Stage
- ResNet18, ResNet50, ResNext50_32x4d를 활용한 Fine-Tuning 진행
- Gradient Accumulation을 활용하여 Batch_size를 증가시킨 후 학습진행(ResNext50_32x4d Batch_size 64 -> NUM_ACCUM 4 -> Batch_size 256)
- Epoch의 변화를 주어 학습
- test set에서의 점수의 급격한 하락으로 인해, Overfitting 의심이 되어 Augmentation 코드화

## 4. 피어 세션
### 4-1. 다양한 시도
- VGG → DenseNet활용하여 성능 증가
- Crop만 진행 → 성능 증가 x
- ResNet50활용 (Parameter의 수를 봤을 때, 적절한 것 같다)
- Crop진행 → Shape의 문제 발생 → Shape이 같아야 Batch를 뽑아올 수 있는 것으로 보이나 어떻게 Shape을 맞춰줘야 하는지에 대해서는 미지수.
- ResNext50_32x4d 활용
- 얼굴 위치에 대한 분포도를 바탕으로 평균, 분산을 활용하여 적당한 Crop 진행 → 얼굴이 해당 위치에 있었으면 했지만 잘 나오지는 않음
- 전처리를 활용하여 주름을 구분해낼 수 있다.(Contrast, Sharpness, Brightness, Color)

### 4-2. 해 볼 시도
- OverSampling
- SMOTE → 직접 적용하기에 어려움이 있지만 시도 해볼 것이다.
- Label이 작은 것을 랜덤 복제하여 Sampling 시도

### 4-3. 질답
- KFold는 Data 수가 너무 적을 때, Validation Set의 Robustness를 주기 위해서 활용하는 방안인데 코드에 직접 입력할 때 성능의 개선이 이루어졌다. 왜 그럴까?
→ 이전 Fold에서의 개선된 Parameter가 다음 Fold로 이어짐으로서 성능 향상을 나타낸다.
- Gradient Accumulation에서 Num_Accum으로 loss를 나누어주게 되는데, 왜 이렇게 하는가?
→ 이유 모색 후 토의

- Gradient Accumulation에서 Num_Accumulation에 대한 내용을 이해가 안된다.
→ Loss를 일반화 시킨다?? 찾아보자!!!

- Brightness & Contrast를 활용하여 배경을 아예 흰색으로 만들 수 있지 않을까?
→ 일정 임계값 이하일 경우, 배경이 검은색이라면 얼굴이 날아갈 위험이 있다.

- Focal Loss 직접 활용해 보았는가?
→ Weight를 직접 조정할 수 있는데 이것이 Focal Loss의 방법과 유사한 것 같다.

- 서버에서 깃헙으로 바로 업로드하는 방법이 있는가?
→ 기존의 방식에서 조금 변화가 있어, 이제는 token을 입력해주면 서버에서 깃헙으로 업로드 할 수 있다.

## 5. 18일차 후기
어느 정도 Baseline Code가 잡혀서 인지 다양한 시도들을 해볼 수 있었다. ResNet18 ,50, DenseNet등에서 실험적으로 성능이 좋게 나오는 것으로 보아 큰 사이즈의 이미지와 연관성이 있는 것도 같고, 왜 성능이 좋게 나오는지를 조금은 더 알아볼 필요성이 있는 것 같다.

더불어, 팀원들의 다양한 시도덕분인지 많은 인사이트를 얻어갈 수 있었고 앞으로 조금더 많은 시도들을 해볼 것 같다!

Accumulation, Focal Loss 등 수업에서 배운 내용 뿐만 아니라 피어세션을 통해 얻은 아이디어를 바탕으로한 Data 전처리 및 Augmentation을 적용해볼 것이다.

이러한 노력 끝에 좋은 결과를 얻을 수 있도록 최선을 다해야겠다. 

## 6. 해야할 일
- Loss 코드 분해해보기
- Focal Loss 적용하기
- Label Smoothing Loss 적용하기
- LR Scheduler 적용해보기
- Gradient Accumulation에서 loss를 NUM_ACCUM으로 나누는 이유는 무엇인가? 학습하기로는 일반화를 해주기 위해서라고 하였는데 이는 무엇을 의미하는가?
