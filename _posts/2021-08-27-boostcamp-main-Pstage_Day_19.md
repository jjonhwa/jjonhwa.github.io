---
layout: post
title: "First P-stage 4(Day 19)"
categories: boostcamp
tags: main
comments: true
---
Ensemble과 Experiment Toolkit에 대하여 학습한다.

**부스트 캠프 19일차 학습 요약**
- **행사** : 오피스 아워
- **학습** : Ensemble, Experiment Toolkit
- **P-Stage** : Augmentation 비율 조정, Augmentation한 데이터 출력
- **피어 세션**

## 목차
- [1. Ensemble](#1-emsemble)
- [2. Experiment Toolkig](#2-experiment-toolkit)
- [3. P-Stage](#3-p-stage)
- [4. 피어 세션](#4-피어-세션)
- [5. 19일차 후기](#5-19일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Ensemble
### 1-1. Overview
- 여러 실험을 하다보면 여러가지 모델로 여러 결과를 만들 수 있다.
- Ensemble의 경우, Production을 베포하는 과정에서 시간이 오래 걸리기 때문에 현업에서 많이 사용하지는 않는다.
- Competition과 같이 성능이 가장 우선시 되는 경우에는 Ensemble의 시도가 아주 좋다.

### 1-2. Ensemble
#### Ensemble
- 싱글 모델보다 더 나은 성능을 위해 서로 다른 여러개의 학습 모델을 사용하는 것
- 모델의 특징을 따져서 조합해야 한다.
- Boosting : 모델을 깊게 계속해서 학습함으로서 제대로 학습하지 못하는 것을 방지할 수 있다. 즉 High Bias의 문제를 어느 정도 해결할 수 있다. 예시로 Gradient Boosting, XGBoost 등이 있다.
- Bagging : 데이터 샘플을 만들어 각각을 학습시킨 모델을 바탕으로 취합해서 평균을 낸다. 이는 각각 다른 모델들로 부터 Samples를 학습시킴으로서 Overfitting을 방지할 수 있는 기법이다. 즉, High Variance 문제를 어느정도 해결할 수 있다. 예로 Random Forest가 있다.

#### Model Averaging (= Voting)
- 각각의 모델들은 답을 내리는 경향성이 다르므로, 그러한 특징들을 바탕으로 여러 모델들을 Ensemble 했을 경우 효과가 있다.
- 서로 다른 모델들은 일반적으로 test set에서 같은 error를 발생시키는 일이 없다는 가정하에서 Model Averaging은 잘 작동한다.
    - Hard Voting : One-Hot으로 구분되어 있는 것을 뜻하며 가장 높은 확률을 가진 값을 1로 표현하는 방법이다.
    - Soft Voting : 각 값들에 대한 확률을 나타내며 Strictly 나타내지 않고 다른 Label에 대해서도 점수를 주는 것이다. 일반적으로 Soft Voting을 많이 사용한다.
#### Cross Validation
- 아까운 Validation Set을 사용할 수 있는 방법이 없을까?라는 질문으로부터 파생된 기법이다.
- 일반화를 어느 정도 개선할 수 있다.
- Fold의 개수에 따라 학습에 사용하는 데이터가 줄어들게 되므로 데이터의 개수가 줄어드는 단점이 있다.
- Boosting의 관점으로 각 폴드별로의 모델들을 Boosting한다고 이해할 수 있다.
- Stratified KFold의 경우 Class의 분포까지 고려한다.

#### TTA (Test Time Augmentation)
- 테스트할 때에 Augmentation을 어떻게 하지?
- Test Set역시 다른 환경에서 다르게 관측될 수 있으므로, 똑같은 Test set을 여러 다른 상태로 만들었을 때, 이를 Train Model에 집어 놓어도 똑같은 결과를 도출할까?라는 질문으로 부터 파생되었다.
- 예를들어, 한 이미지에 대한 Noise 섞인 5개의 이미지가 있다고 가정할 때, 이들 각각의 확률(Soft Voting)을 계산해서 이들을 Averaging하여 최종 output을 결정한다.

#### 성능과 시간의 trade off
- 앙상블의 효과는 확실히 있지만, 그만큼 학습, 추론 시간이 배로 소모된다.
- Production 환경에서 사용되는 모델의 경우 성능보다는 효율이 중요하므로 앙상블을 하지 않는 경우가 많다.
- Score를 올리는 것이 중요할 경우에는 사용하는 것이 좋다.

### 1-3. Hyper-Parameter Optimization
- 앙상블 만큼이나 많은 시간이 소모된다.
- 효과가 엄청나게 좋거나 하지는 않는다.
-
#### Hyper Parameter
- 시스템의 메커니즘에 영향을 주는 주요한 Parameter이다.
    - Learning Rate
    - Hidden Layer의 개수
    - Batch_size
    - Loss
    - Optimizer
    - KFold에서 K의 개수
    - Dropout / Regularization 등이 있다.
- Hyper Parameter가 하나하나 변할 때마다 다시 학습해야하므로 시간이 압도적으로 많이 할애된다.
- 시간과 장비가 충분하다면 해볼만 하지만 일반적인 딥러닝을 할 경우에는 많이 사용하지는 않는다.
- Bayesian Optimization 기법이 가장 많이 사용된다.
    - 가볍게 얘기하자면, Bayesian Optimization은 가상의 함수를 만들고 그 함수의 Bound를 줄여나가면서 가장 최적의 과정을 찾는 방식이라고 이해할 수 있다.
    - 가장 많이 쓰이는 기법이다.
- Optuna를 활용할 수 있으며 다음과 같이 활용할 수 있다.
![1주차 14](https://user-images.githubusercontent.com/53552847/131125508-4378685c-bd20-4efa-92f4-5d16b1b58bac.PNG)

## 2.Experiment Toolkig
### 2-1. Tensorboard
- 기존에는 Tensorflow에서만 사용되다가 최근에 PyTorch로 포팅되어 사용할 수 있게 되었다.
![1주차 15](https://user-images.githubusercontent.com/53552847/131125514-26d69e93-e81b-4ae7-8431-8cd9301083c0.PNG)
![1주차 16](https://user-images.githubusercontent.com/53552847/131125516-e16c3aff-0504-4d25-8c0f-b307dc3965be.PNG)
- 위의 그림들과 같이 학습의 과정을 살펴볼 수도 있고, 어떤 이미지들이 Train되었고 Validation 되었는지도 확인할 수 있다.
- 사용법은 다음과 같다.
![1주차 17](https://user-images.githubusercontent.com/53552847/131125519-b3a1e471-9f11-4f8d-b8e4-a29e64bc8437.PNG)
- Tensorboard의 경우 webpage에서 실행이 되며 일반적으로 내 컴퓨터에서 사용할 경우 host ADDR은 필요없고 서버에서 사용할 경우 host ADDR만 0.0.0.0을 기입해서 열려있는 서버의 포트번호를 입력해 사용할 수 있다.

### 2-2. Weight and Biad (wandb)
- 딥러닝 로그의 깃허브 같은 느낌으로 구성되어있다.
- 컴퓨터와 wandb를 연결해서 사용한다.
- Tensorboard 보다 더 간단하다..
- 사용방법은 다음과 같이 wandb init, log를 설정하여 사용할 수 있으며, 특히 Python Project를 진행할 때 사용하면 보다 편리하게 진행할 수 있다.
![1주차 18](https://user-images.githubusercontent.com/53552847/131125521-9cbd899c-0ad5-43cd-9c4c-ed29da7be299.PNG)
![1주차 19](https://user-images.githubusercontent.com/53552847/131125523-d461a685-6a78-4831-8fa0-54ae3d249e80.PNG)

### 2-3. Machine Learning Project
- Jupyter Notebook
    - 코드를 아주 빠르게 Cell 단위로 실행해볼 수 있는 장점이 있다.
    - EDA를 진행할 때에 매우 편리하다.
    - 전처리 작업에 대한 결과를 바로 재사용할 수 있다.
    - 하지만, 노트북 창이 꺼지거나 원격 진행에서 발생하는 불안정성의 문제가 있다.
- Python IDLE
    - 구현은 한번, 사용은 언제든, 간편한 코드 재사용
    - 코드를 패키지 형태로 우리가 직접 구현해놓고 필요할 때마다 Import 하여 사용할 수 있다.
    - 시간적 이득이 크다.
    - 디버깅 측면해서 어떤 코드든 무섭지 않게 만들 수 있다.
    - 실험 핸들링이 쉽다. -> Configuration 핸들링을 통한 손쉬운 실험이 가능하다.
    
### 2-4. Some Tips
- 다른 사람의 코드를 볼 때에 분석 코드보다는 설명글을 유심히 보자.
- 필자의 생각의 흐름을 파악하고, 왜 이걸 했는지, 앞으로는 무엇을 할 것인지를 파악하자.
- 코드는 사람마다 다르므로, 그 코드의 디테일한 부분까지 이해하고 넘어가도록 하자.
- Paper with Codes -> 최신 논문과 코드가 함께 올라오는 사이트이므로 참고하도록 하자.
- 공유하는 것을 주저하지 말고 새로운 배움의 기회가 될 수 있으므로 공유하는 버릇을 들이자.

## 3. P-Stage
- ResNext50_32x4d를 활용
- EDA를 바탕으로 각 class별 이미지의 개수를 Augmentation을 적용하여 보완해주었다.
- Augmentation의 경우 다양성이 확보되지 않은 채로 불리기만 하면 오히려 성능에 안좋은 영향을 미칠 수 있다. -> Overfitting이 될 수 있는 것 같다.
- Dataset에서 Augmentation을 진행해주었으며 Generation 형태로 입력된 이미지를 출력해서 어떤 식으로 Augmenation이 진행되었는지 확인하였다. -> Tensorboard 혹은 Ray를 이용한다면 바로바로 Train set과 Valid set을 확인할 수 있다.

## 4. 피어 세션
- 추후 업로드 

## 5. 19일차 후기
Augmentation을 날코딩을 통해 어떻게 어떻게 구현에 성공하였다. 물론 Code Refactoring을 진행하고나 Class로 불러올 수 있도록 조만간 수정을 진행해야 겠지만 이론을 바탕으로 구현을 해냈다는 것에 대하여 만족스러웠다.

별거 아닌 내용이지만 중간 중간에 많은 시행착오가 있었는데 이들을 해결하고 최종 모델 학습 및 Augmentation 데이터를 직접 확인해 봄으로서 어느 정도의 진도를 나갈 수 있었던 것 같다.

물론, 강의 및 오피스아워를 통해 내가 한 것들은 정말 뻘짓이였구나 싶은 감은 있지만 헛된 것이라고 생각하지말고 이해하고 구현한 내용을 바탕으로 High Level의 구현도구들을 사용해보도록 하자

주말동안 많은 시도들을 해보고 좋은 성적 거둘 수 있도록 최선을 다하자!

## 6. 해야할 일
- Wandb 혹은 Tensorboard 적용해보기
- TTA 방법 확인하기
- Bayesian Optimization 공부하기
- Python IDLE을 활용하여 실험해보기
