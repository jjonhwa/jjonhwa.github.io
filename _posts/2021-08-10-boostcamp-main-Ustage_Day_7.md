---
layout: post
title: "Ustage Day 7"
categories: boostcamp
tags: main
use_math: true
comments: true
---
Neural Network와 딥러닝 기본 용어에 대한 설명을 한다.

**부스트 캠프 7일차 학습 요약**
- **행사** : 도메인 특강, 마스터 클래스(안수빈) 
- **학습** : Optimization
- **피어 세션**

## 목차
- [1. Importance Concepts in Optimization](#1-importance-concepts-in-optimization)
- [2. Practical Gradient Descent Method](#2-practical-gradient-descent-method)
- [3. 피어 세션](#3-피어-세션)
- [4. 7일차 후기](#4-7일차-후기)
- [5. 해야할 일](#5-해야할-일)

## 1. Importance Concepts in Optimization
### 1-1. Optimization
- 최적화와 관련된 주요 용어
- 다양한 Gradient Descent 기법이 존재
- Generalization, Overfitting, Cross-Validation 등의 용어들에 대하여 설명한다.
- 기존 SGD를 넘어 최적화가 더 잘될 수 있도록 다양한 기법들에 대해 배운다.
- Gradient Descent 기법에 따른 성능변화를 알아본다.
- 실질적으로 써먹을 때 중요하게 알아두면 좋을 Concept들에 대하여 설명한다.
- 용어 대한 정의를 제대로 짚고 넘어가지 않으면 많은 오해를 불러 일으킬 수 있다.

### 1-2. Gradient Descent
- 미분을 활용한 Local Minimum을 찾을 수 있는 1차원 미분 반복활용 최적 알고리즘(First-Order Iterative Optimization)

### 1-3. Important Concepts in Optimization
- Generalization
- Under-fitting vs Overfitting
- Cross Validation
- Bias-Variance Trade-Off
- Boostrapping
- Bagging & Boosting

### 1-4. Generalization
- 흔히, 일반화 성능을 높이는 것이 우리의 목적(새로운 데이터에 대하여 좋은 성능을 내는 것)
- 일반적으로, Training Error와 Test Error의 차이를 이야기한다. (좋은 Generalization Performance를 가지고 있다는 것은 이 Network의 성능이 학습데이터와 비슷하게 나온다는 의미이다.)
- Generalization이 좋다고 해도, 일반적인 Training의 성능이 좋지 않을 수 있어서 무조건적으로 좋은 것은 아니다.

### 1-5. Cross-Validation
- Model을 Test Dataset에 독립적으로 일반화 시키기 위하여 사용하는 Model Validation Technique이다.
- N, Hyperparameter(Learning Rate, Optimizer)를 구하게 되는데, 이  Cross-Validation을 활용하여 최적의 Hyperparameter Set을 찾고 이를 고정한 후에 모든 데이터를 활용하여 학습을 진행한다.
- Parameter : 내가 최적해에서 찾고 싶은 값(Weight, Bias)
- Test 데이터는 어떤 식으로든 학습할 때 사용되어서는 안된다.

### 1-6. Bias & Variance
- Variance : 비슷한 입력을 넣었을 때, 출력이 얼마나 일관적으로 나오는가를 의미하며, Variance가 높을 경우 Overfitting의 위험이 있다.
- Bias : 평균적으로 봤을 때, 값들이 True Target에 가깝게 되면 Bias가 낮음을 의미하고 Mean에서 많이 벗어나 있을 경우 Bias가 높다.
- Variance-Bias-Trade-off :
![DL Basic 5qjs](https://user-images.githubusercontent.com/53552847/128871762-1d1bf41e-5a31-4b6d-840d-7a7548e5da24.PNG)
- 위의 그림처럼 Loss는 Variance, Bias, Noise로 나눌 수 있으므로 Variance가 낮아짐에 따라, Bias는 높아질 수밖에 없다.

### 1-7. Bootstrapping
- 학습 데이터가 고정되어 있을 때, Sub-Sampling을 통해 학습데이터를 여러 개로 만들고, 이를 활용하여 여러 모델을 만들어 활용한다.

### 1-8. Bagging vs Boosting
#### Bagging
- Bootsstrapping Aggregating
- 학습 데이터를 여러개로 쪼개서 여러 모델을 만들고, 여러 모델들의 성능의 평균을 낸다. (흔히, Ensemble이라고 한다,)
- 사실, 전체를 활용한 한 개의 모델보다 여러개로 쪼개서 여러개의 모델을 Voting 혹은 Averaging 등을 활용한 출력을 활용하는 것이 더 좋은 성능을 내는 경우가 많다.

#### Boosting
- 학습 데이터를 Sequential하게 바라봐서, 간단한 모델을 만들고, 이 모델을 학습데이터에 대하여 돌려보고 제대로 예측하지 못한 데이터에 대하여 다시 잘 작동하는 모델을 만드는 방식으로 반복한다.
- 이렇게 해서 만들어진 여러개의 모델을 합쳐서 활용하는 것이다.
- 각각의 모델들(Weak Learner)을 Sequential하게 합쳐서 하나의 Strong Learner를 만드는 것이다.
- 각 Weak Learner들의 Weight를 찾는 식으로 정보를 취합하게 된다.
- Bagging은 Parallel, Boosting은 Sequential의 느낌으로 이해하면 쉽다.

## 2. Practical Gradient Descent Method
### 2-1. Batch-size Matters
- Batch-Size를 결정하는 것은 굉장히 중요한 문제이다.
- Large Batch를 사용하게되면 Sharp Minimizer에 도달한다.
- Small Batch를 사용하게되면 Flat Minimizer에 도달한다.
- 일반적으로 조금 작은 Batch Size를 활용한다 -> Sharp Minimizer보다는 Flat Minimizer에 도달하는 것이 좋다.
![DL Basic 6번](https://user-images.githubusercontent.com/53552847/128871765-ec2bfe3b-1945-4c68-b1e3-79f31e759362.PNG)
- 위의 그림과 같이 Train, Test가 있다고 할 때, Flat의 경우 train, test 사이의 차이가 크지 않지만 Sharp의 경우 엄청난 차이가 발생한다. 즉, Sharp Minimizer에 도달했을 경우 Generalization Performance가 떨어짐을 알 수 있다.

### 2-2. Loss를 최소화하기 위해서 어떤 Optimizer를 선정해야하는가?
#### Stochastic Gradient Descent
- Gradient를 자동으로 구해준다.
- Learning Rate를 구하는 것이 어렵다.
- 초기에 등장한 Optimizer이며 이를 바탕으로, 그리고 이에 더해 '어떻게 빨리 학습시킬 수 있을까?'라는 질문으로부터 다른 많은 Optimizer들의 파생을 이끌었다.
![DL Basic 7번](https://user-images.githubusercontent.com/53552847/128871768-2be00fa2-5cc0-41ab-88d1-b06bab6e519a.PNG)

#### Momentum
- '이전 Gradient가 작동한 방향을 다음 Gradient에 활용해보자'에서 출발
- 이전 Gradient에 Momentum이 포함된 Gradient로 업데이트한다.
- Gradient가 이쪽 저쪽으로 흐른다 할지라도, 어느 정도 잘 학습하게 한다.
- Converge를 못하게 되는 현상이 발생한다.
![DL Basic 8번](https://user-images.githubusercontent.com/53552847/128871769-7e2784cb-9eaa-40cc-80fb-efe94ccd14a0.PNG)

#### Nesterov Accelerate Gradient
- Momentum과 비슷하다.
- 이전 Gradient를 활용할 떄, Lookahead Gradient로 적용한다.
- Lookahead Gradient : 한 번 이동한다. 구해진 방향으로 한 번 진행한 후의 Gradient를 활용
- 좀 더 빨리 Converge하는 효과가 있다.
![DL Basic 9번](https://user-images.githubusercontent.com/53552847/128871772-d29dbe33-13db-40a1-85a7-63f561146479.PNG)

#### Adagrad
- 지금까지 Weight이 얼마나 변했는지 혹은 안 변했는지를 활용한다.
- 많이 변했으면 조금은 적게, 적게 변했으면 조금은 많게 변하게 한다.
![DL Basic 10번](https://user-images.githubusercontent.com/53552847/128871774-c1705836-febe-407d-ad28-e5b3fce00593.PNG)
- $$G_t$$ : Sum of Gradient Squares. 지금까지 Gradient가 얼마나 많이 변했는지를 제곱해서 합한 것이다.
- Epsilon : 0으로 나누는 것을 방지하기 위한 아주 작은 값
- 문제 : G는 계속해서 커진다. 결국 G가 무한대로 간다면 W의 업데이트가 되지 않는다. 즉, time step이 흘러갈 수록 학습이 멈추는 현상이 발생한다.

### Adadelta
- Adagrad의 $G_t$가 커지는 현상을 막기 위해 등장하였다.
- Window Size 시간에 대한 Gradient 제곱을 확인한다.
![DL Basic 11번](https://user-images.githubusercontent.com/53552847/128871778-5189a85b-3c06-4a64-86cd-595fcdac903a.PNG)
- 문제 : Window Size를 100이라고 했을 때, 이전 100개에 대한 Graident의 정보를 가지고 있어야한다. 하지만 만약 Gradient의 사이즈가 너무 크게 된다면 GPU가 터지는 현상을 초래한다.
- 이를 보완하기 위해서 사용되는 것이 위 식에서의 $$\gamma$$ 이다. 마지막 식을 활용하게 되면 100개의 윈도우에 대한 평균값만을 보유하게 된다.
- Learning Rate가 없다. 바꿀 수 있는 요소가 많이 없기 때문에 많이 사용되지 않는다.

#### RMSprop
- Adagrad에서 $G_t$에 EMA(Exponential Moving Average)를 더해줘서 사용하게 된다.
![DL Basic 12번](https://user-images.githubusercontent.com/53552847/128871782-36ddd30b-9166-4272-bf4a-ea1e9928fa88.PNG)

#### Adam
- Gradient Squares를 EMA를 활용해 가져감과 동시에 Momentum을 함꼐 활용한다.
![DL Basic 13번](https://user-images.githubusercontent.com/53552847/128871783-07f01a45-76da-4f4a-bd2e-38c356324909.PNG)
- 마지막 $\beta$ term은 전체 방법론(전체 Gradient Descent)이 Unbiased Estimator가 되기 위해 수학적인 증명을 활용한 것이다. (깊게 들어갈 필요 x)
- Epsilon은 보통 $$1e^{-7}$$ 로 정해져 있긴 하지만 이를 잘 설정해주는 것 역시 Practical 하게 중요하다.

### 2-3. Regularization
- 규제를 건다. 학습을 반대, 방해하도록 규제한다.
- 학습을 방해함으로서 학습 데이터에만 잘 작동하는 것이 아니라 테스트 데이터에서도 잘 동작하게 만들어준다.

#### Early Stopping
- 학습을 일찍 멈추는 것
- Training Error는 계속해서 줄어들지 몰라소 Test Error는 어떤 기점을 기준으로 다시 상승하기 시작한다. 이 떄, 어떤 특정 지점을 선택하여 학습을 중단한다.
- Validation Error를 활용하여 먼저 멈춘다.

#### Parameter Norm Penalty
- Neural Network의 Parameter가 너무 커지지 않게 한다.
- 함수의 공간 속에서 함수를 최대한 부드럽게 한다. (여기서 함수가 부드럽다는 것은 Generalizaiton Performance가 좋다라는 가정이 존재한다.)
![DL Basic 14번](https://user-images.githubusercontent.com/53552847/128871786-3acfe04b-6e4e-4cda-9004-23663425e7cf.PNG)

#### Data Augmentation
- 데이터가 많으면 많을수록 학습은 더 잘된다.
- Dataset이 적다면 Traditional ML이 더 좋을 수 있다.
- Dataset이 어느 정도 커지게 되면 Traditional ML로는 많은 데이터들을 표현하는 표현력이 떨어진다.
- 많은 Dataset이 필요하지만 직접 Dataset을 만들 수 없기 때문에 Data Augmentation을 활용한다.
- 데이터를 회전, 반전 등으로 Label이 변하지 않는 선에서 데이터를 변형시켜 이를 데이터셋으로 활용한다.(6을 회전시켜 label이 9로 변하는 것과 같은 형태로는 사용해서는 안된다.)

#### Noise Robustness
- 왜 잘 작동하는지에 대해서는 아직 의문
- 입력데이터에 Noise를 집어 넣는다. 단순히, noise를 input에만 넣는 것이 아니라 Weight에도 섞어준다.
- 학습시킬 때, Noise를 활용해서 Weight을 흔들어주면 성능이 더 잘나온다. -> 실험적인 결과

#### Label Smoothing
- Data Augmentation과 비슷
- 학습 데이터 2개를 뽑아서 섞는 형태
- Mixup(사진 겹치기), Cutout(사진 일부 삭제), CutMix(사진 일부분을 다른 사진으로 대체)
- 분류문제를 해결 중이고, 데이터 셋이 한정적일 경우 Dataset을 더 얻을 수 있는 방법이 없다면 Mixup 혹은 CutMix를 활용해보는 것은 정말 좋다.
- 노력대비 좋은 성능을 얻을 수 있는 방법론이다.

#### Dropout
- 중간 중간의 Weight을 0으로 바꿔주는 것
- 각 뉴런들이 Robust한 Feature들을 잡아줄 수 있다고 해석만 한다.
- 일반적으로 사용하면 성능은 증가한다.

#### Batch Normalization
- 논란이 많은 연구
- BN을 적용하고자 하는 Layer의 Statistics를 정교화시키는 것
- 일반적으로 활용하면 성능이 증가한다.
- 일반적인 분류문제를 해결할 때 성능을 올릴 수 있다.



## 4. 피어 세션
### 4-1. [지난 질문들]

- SVD 특이값 분해
Q. SVD에서 Singular value가 원래 크기 순으로 정렬되어 있는가?
- 맞습니다.

  
### 4-2. [강의 관련 질문]

Q. Adadelta - 실제로 거의 사용하지 않는 이유는 성능이 낮기 때문인가요?
- 성능이 낮다기 보다 learning rate가 없어 최적화가 관여할 요인이 없다.
- 다른 optimizer가 일반적으로 학습에 관여할 부분이 많아 성능이 더 잘나온다.


### 4-3. [선택과제 1번 ViT 관련 질문]

Q. Transformer encoder 구현할 때 residual 부분이 다른 코드와 비교했을때, class로 구현하지 않더라도 맞나요?
- 맞습니다.

Q. attention visualization에서 error가 납니다.
- slack에서 다른 캠퍼분께서 수정하는 부분을 알려주셨다.
- 수정하지 않고도 attention을 리스트에 append하면서 shape을 맞추었더니 작동했다.

Q. position embedding을 할때 random한 값을 넣는 이유?
- 학습하는 값으로 생각해 랜덤하게 생성한 후 더해주었다.

Q. cls_token과의 concat으로 x의 shape이 어떻게 바뀌는 건가요?
- x의 shape이 (1, 49, 16)에서 concat으로 (1, 50, 16)이 된다.
- 이후 position의 shape이 (50, 16)이므로 x와 position을 더하는 과정에서 broadcasting이 일어난다.

Q. nn.linear mlp와의 차이가 없는건가요?
- nn.linear는 layer 그 자체로 fully connected된 선형변환이라고 생각.
- mlp는 activation function이 들어갔다.

Q. colab과 구글드라이브에서 데이터 저장이 마운트 없이 가능한가요?
- 가능하지만 구글드라이브에 저장하면 다운로드도 가능하다.

해결하지 못한 질문들은 대부분 아직 배우지않은 모델에서 나왔습니다.
이후 멘토님과의 시간에서 질문하기로 했습니다.

Q. attention을 넣을때 list를 만들어 넣어도 잘 작동하는 이유는 무엇인가요?

Q. transformer에 들어가는 latent vector는 각각의 patch를 의미하나요?

Q. self-attention의 shape이 768 -> 64 ->768로 바뀌는데 어떻게 되는건가요?


## 5. 7일차 후기
도메인 특강을 통한 NLP, CV에 대한 카테고리 및 진학내용에 대한 내용을 들었는데 내용을 자세히 설명해주셔서 보다 도움이 되었던 특가이었던 것 같다. 몰랐던 내용을 소개해주기도 하고 앞으로 공부 방향성에 대해서도 어느정도 정해줬던 강의였기에  조금은 앞으로의 공부의 방향성을 정할 수 있게되었다. 

아직 확실한 방향성은 잡지 못했지만 오늘 특강을 토대로 조만간 방향을 결정지을 수 있을 것 같다.

더불어, 안수님 캐글 그랜드 마스터님의 시각화 특강이 있었는데 아니나 다를까 꽤나 흥미로운 내용을 말씀해주셨다. 아직은 현업을 접해보지 못해 시각화의 필요성을 깊게 느끼진 못했지만 현업에서의 중요성을 알게되었고 언젠가는 분명히 해야한다는 것을 알 수 있었다.

## 6. 해야할 일
- Further Question > 올바르게 Cross-Validation을 하기 위해서는 어떤 방법들이 존재하는가?
- Further Question > Time Series의 경우 일반적인 K-Fold CV를 사용해도 되는가?
- Momentum : Momentum항의 의미, Converge를 왜 못하게 하는지
- Adadelta에 대한 내용 + '바꿀 수 있는 요소가 많이 없기 때문에 많이 사용되지 않는다.'라는 것이 사용하지 않는 이유가 될 수 있는가?
- RMSprop, Adam에 대한 내용 보충
- Batch Normalization에 대한 내용 보충
