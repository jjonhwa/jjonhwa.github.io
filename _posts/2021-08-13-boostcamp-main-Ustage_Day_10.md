---
layout: post
title: "Ustage Day 10"
categories: boostcamp
tags: main
comments: true
---
Generative Model에 대한 설명을 진행한다.

**부스트 캠프 10일차 학습 요약**
- **행사** : 마스터 클래스, 스페셜 피어세션
- **학습** : Generative Model 1, Generative Model 2
- **피어 세션**

## 목차
- [1. Generative Model 1](#1-generative-model-1)
- [2. Generative Model 2](#2-generative-model-2)
- [3. 피어 세션](#3-피어-세션)
- [4. 10일차 후기](#4-10일차-후기)
- [5. 해야할 일](#5-해야할-일)

## 1. Generative Model 1
### 1-1. Introduction
- Generation : 새로운 이미지를 찍어낼 수 있다.
- Density Estimation : 입력 이미지 x에 대한 확률값으로부터 분류(강아지 인지, 고양이인지)를 구분해낼 수 있다. (Anomaly Detection에 활용)
- Generative Model은 Discriminative Model을 포함한다. -> 새로운 데이터를 만드는 것 뿐만 아니라 분류 모델로서의 역할 역시 해낼 수 있다.
- Explicit Model이다. -> 입력에 대한 확률값을 얻어낼 수 있는 모델이다.
- Unsupervised Representation Learning(Feature Learning) : 이미지의 특성을 학습할 수 있다. (하지만, 이 부분에 대해서는 조금 의하한 부분이 있다.)

### 1-2. Basic Discrete Distributions
#### Bernoulli Distribution
![DL Basic 40](https://user-images.githubusercontent.com/53552847/129351653-f763eea5-9fc4-457e-9152-97736e4a4259.PNG)

#### Categorical Distribution
![DL Basic 41](https://user-images.githubusercontent.com/53552847/129351657-6911089e-d4cb-4f17-be8b-a99ba67d57e9.PNG)

### 1-3. Structure Through Independence
- 이하에서의 설명은 Binary Distribution을 활용한다.
- Binary Distribution을 활용하게 될 때, Parameter의 개수는 $$2^n$$이 된다.
- Possible State는 $$2^n$$으로 동일하지만 사용되는 Parameter의 수가 달라진다.
- 만약, 입력 X가 모두 독립임을 가정한다면 Parameter의 수가 n개로 줄어든다. 하지만, 이 가정은 성립할 수 없다.
- Fully(전부 다 Dependent하다고 할 때)는 Parameter가 너무 많고, Independent는 Parameter가 적지만 불가능한 가정이다.그렇다면 그 사이는 어떻게 구할 수 있을까?

### 1-4. Conditional Independence
- 3개의 중요한 Rules :
![DL Basic 42](https://user-images.githubusercontent.com/53552847/129351660-4bd774cb-6617-45e4-88b6-c69cf78e503c.PNG)
    - Chain, Bayes' Rule은 Exact한 방법이다.
    - Conditional Independence는 가정이다.
    - Chain Rule과 Conditional Independence를 섞어서 잘 사용해주면 Fully Model과 Independence Model 사이의 좋은 모델을 만들 수 있다.

- Chain Rule을 통해서 Joint Distribution을 Conditinoal Distribution으로 표현할 수 있다. -> 이 때, Parameter의 개수는 $$2^n - 1$$ 동일하다. 왜냐하면 Chain Rule은 식을 변현해준 것일 뿐이기 때문이다.
- Markov Assumption 가정(Conditional Independence 중 바로 이전만 의존) : 바로 전에 들어오는 값만 의존적이고 나머지 값에는 독립임을 가정한다. 즉, $$X_10$$는 $$X_9$$에만 Dependent하고 나머지 $$X_1$$ ~ $$X_8$$과는 Independent함을 의미한다.
- Chain Rule에 대한 식이 Markov Assumption 가정에 의해 다음과 같이 변한다. (Markov Assumption 가정은 Conditional Independence에 내용을 보여주기 위한 예시이다.)
![DL Basic 42-1](https://user-images.githubusercontent.com/53552847/129354576-8f1c8040-fb27-4fd9-8d2e-fbd39f815e5f.png)

- 위와 같은 그림으로 변할 때, Parameter의 개수는 2n-1개가 된다. (-1을 해주는 것은 자유도 때문이다.)
- 중간에 등장하는 Conditional Independence를 어떻게 주냐에 따라서 Parameter의 수를 잘 바꿀 수 있다.
- 위와 같은 방법을 Auto-Regressive Model이라 부르고, Conditional Independency를 잘 이용한 것이고 AR Model이라고 부른다.

### 1-5. Auto-Regressive Model
- 이정 정보들에 Dependent한 Model을 의미한다.
- 즉, 바로 이전에만 의존적이어도, 전부 다 혹은 몇 개에만 의존적이어도 AR Model이라고 한다.
- AR Model은 Order를 잘 매김하는 것이 중요하다. 더불어, Image에서의 순서를 매김하는 것은 애매하기 때문에 어떻게 순서를 매김하냐에 따라 성능 및 방법론이 달라질 수 있다.
- 흔히, 이전 N개만 고려하는 모델을 AR-N Model 이라고 하고, Markov Assumption Model은 AR-1 Model이다.
- 어떤 식으로 Conditional Independence를 주냐에 따라 전체 모델의 Structure가 달라지게 된다.

### 1-6. NADE : Neural Autoregressive Density Estimator
- 기본적인 AR Model로 현재의 데이터가 과거의 모든 정보에 Dependent한 모델이다.
![DL Basic 43](https://user-images.githubusercontent.com/53552847/129351661-bc50ffb3-aa7d-4f0b-b909-ea494366cb83.PNG)

- Nerual Network 입장에서는 입력 차원이 계속 달라지게 됨으로 Weight 역시 계속해서 달리지게 된다.
- NADE는 Explicit한 모델이다. -> 임의의 입력이 주어지면 이들의 확률을 계산할 수 있다.
- 입력값에 대한 조건부 확률값을 Chain Rule 식에 대입함으로서 최종 확률을 구할 수 있다. 이로부터, 단순히 Generation만 할 수 있는 것이 아닌 어떤 입력에 대한 확률을 구할 수 있는 Explicit한 모델이다.
- Density Estimator는 Explicit Model을 의미하는 경우가 많다. (확률을 Explicit하게 계산할 수 있음을 의미한다.)
- 이와 반대로 Implicit Model은 단순히 Generation만 할 수 있다.
- 위에서 활용한 것은 Discrete Model이었으며, Continuous Model은 마지막 Layer에 Gaussian Mixture Model을 활용하여 나타낼 수 있다.

### 1-7. Pixel RNN
![DL Basic 44](https://user-images.githubusercontent.com/53552847/129351664-61f87fb7-6b9c-4e86-bdaa-b0849e5bed3c.PNG)
- 이미지에 있는 Pixel들을 만들어 내고 싶은 것.
- RNN을 통해서 Generation하겠다는 아이디어.
- Ordering을 어떻게 하느냐에 따라 Row LSTM, Diagonal BiLSTM으로 나뉜다.
![DL Basic 45](https://user-images.githubusercontent.com/53552847/129351634-5b10d16f-e0f8-48d4-bc0e-cf9add10cea5.PNG)
- Row LSTM : $$i_{th}$$ 픽셀을 만들 때, 위쪽의 정보를 활용한다.
- Diagonal BiLSTM : 이전 정보를 모두 활용.

## 2. Geneartive Model 2
**참고 : 이 파트에서의 오류가 있으므로 충분히 공부한 후 내용을 다시 수정하도록 하자!**
- Latent Variable Model -> "Variational Inference and Deep Learning: A New Synthesis" 논문 읽어보기

### 2-1. Variational Auto Encoder (VAE)
#### Variational Inference (VI)
- Posterior Distribution을 찾는 것이 목적이다. 
- Posterior Distribution : 나의 Observation이 주어졌을 떄, 내가 관심있어 하는 Random Variable의 확률분포이다.
![DL Basic 46](https://user-images.githubusercontent.com/53552847/129351637-ade356df-4bdc-493b-b9cf-d17519d367d2.PNG)
- 위의 식에서 z와 x가 반대로 되는 것을 Likelihood라고 부른다.
- 여기서 z는 Latent Vector가 된다.
- 일반적으로 Posterior Distribution을 계산하는 것은 매우 어렵다. -> 우리가 학습할 수 있는 혹은 최적화할 수 있는 어떤 것으로 근사하겠다는 목적으로서 이에 근사하는 분포가 다음과 같은 Variational Distribution이다.
![DL Basic 47](https://user-images.githubusercontent.com/53552847/129351638-b06b36d4-5852-4a84-8015-f9905f531600.PNG)
- 내가 찾고자 하는 Posterior Distribution에 제일 잘 근사할 수 있는 Variational Distribution을 찾는 이 일련의 과정을 Variational Inference라고 부른다.
- 무언가를 잘 찾겠다. 무언가를 잘 최적화하겠다라고 할 때, 우리에게 필요한 것은 Objective, 즉 Loss Function이다.
- VI에서는 KL Divergence라는 Metric을 활용해서 내 Variational Distribution과 Posterior Distribution을 줄이도록 한다.
- Posterior가 뭔지도 모르는데 어떻게 이를 근사하는 Variational Distribution을 구할 수 있을까? -> 다음과 같은 ELBO Trick을 사용한다.
![DL Basic 48](https://user-images.githubusercontent.com/53552847/129351639-80563e8c-cc1b-4310-9c9d-87d1ba102929.PNG)
- 궁극적으로 KL Divergence를 줄이는 것이 목표인데, 이것이 불가능 하므로, ELBO(Evidence Lower Bound)를 증가시킴으로서 반대극부로 우리가 원하는 것을 얻을 수 있다.
- 이것이 모두 Variational Inference를 의미한다.(ELBO를 Maximize함으로부터 얻을 수 있다.)

#### ELBO (Evidence Lower Bound)
![DL Basic 49](https://user-images.githubusercontent.com/53552847/129351641-a22a19b3-bf8a-4c01-9afc-e91f14432284.PNG)
- ELBO를 나눠보면 위와 같은 식이 나온다.
- Reconstruction Term과 Prior Fitting Term으로 나오는데, Reconstruction Term이 Auto-Encoder에서의 Reconstruction Loss를 최소화 해준다.
- Reconstruction Term : Encoder를 통해 X를 Latent Space로 보냈다가, 다시 Decoder로 돌아오는 Reconstruction Loss를 줄여주는 term이다.
- Prior Fitting Term : X라는 이미지가 여럿 있다고 할 때, 이 이미지들을 Latent Space로 올려놓는다. 그러면 이 이미지들은 Latent Space에서의 점이 되는데, 이 점들이 이루는 어떤 분포가 이 Latent Space에서 내가 가정하는 Prior distribution과 비슷하게 만들어준다.
- 위의 이유 때문에, Variational Auto-Encoder가 Generative Model이 될 수 있다.

#### Variational Auto Encoder의 의미
- 어떤 입력이 주어지고, 이를 Latent Space로 보내서 무언가를 찾고, 이를 다시 Reconstructure하는 Term으로 만들어주는데, Generative Model이 되기 위해서는 Latent Space가 된 Priror Distribution으로 Data Sampling을 하고 이를 Decoder로 태워서 나오는 어떤 이미지 혹은 Output Domain의 값을 우리의 Generation Result로 보는 것이다.
- Auto Encdoer는 단순히 Input이 Latent Space로 갔다가, Output이 바로 나오므로 Generative Model이 아니다.
- Key Limitation :
    - VAE는 Explicit한 Model이 아니다. -> 어떤 입력이 주어졌을 때, 얼마나 Likely한지, 즉 그럴싸한지 알기가 어렵다. 그래서 Intractable Model이라고 한다.
    - ELBO가 Reconstruction Term과 KL Divergence Term(Prior Fitting Term)으로 나누어 지는데 Reconstruction Term의 경우 바로 Neural Network를 태우면 된다. 하지만, KL Divergence는 Gaussian을 제외하고는 Float term(수정 - 정확하지 않다.)이 나오는 것이 많이 없다. 즉, Prior Fitting Term은 미분가능해야하는데 적분이 Intractable하게 되면 미분이 불가능하다.
    - 대부분에서 Isotropic Gaussian을 활용하게 된다. -> 모든 Output Dimension이 Independent한 Gaussian Distributiond을 의미한다.
![DL Basic 50](https://user-images.githubusercontent.com/53552847/129351642-c934e5e9-bc76-4cdb-8039-8a2ac90a10d2.PNG)
- VAE는 이미지, 문장 등을 만들 수 있다.

### 2-2. Adversarial Auto Encoder (AAE)
- GAN을 활용해서 Latent Distribution 사이의 분포를 맞춰준다.
- VAE에서 Prior Fitting Term을 GAN Object Term으로 바꾼 것이다.
- 그래서 AAE는 Latent Distribution을 Sampling만 가능한 어떤 분포만 이떠라도 이것과 맞출 수 있다. (수정 -> 이것이 뭐지?)
- 복잡하고 다양한 분포들을 우리가 Latent Private Distribution으로 활용할 수 있다는 것이 장점이다.
- Latent Distribution 사이의 워스슈타인 디스턴스(수정 -> 무슨 말이지?)를 줄여주는 것과 똑같은 효과가 있다.
- 연구에 실질적으로 많이 쓰인다.
- Generative Quality도 VAE에 비해서 훨씬 좋은 경우가 많다.

### 2-3. Generative Adversarial Network
![DL Basic 51](https://user-images.githubusercontent.com/53552847/129351644-1828a27a-02c6-4938-9bfa-0935d1046d4c.PNG)
- 학습의 결과로 나오는 Generator를 학습하는 Discriminator가 점차 좋아진다.
- Explicit Model이다.
- Two Player minimax game between generatro and discriminator이다.
    - minimax game : 한 쪽은 높이고 싶어하고, 한 쪽은 낮추고 싶어하는 게임
![DL Basic 52](https://user-images.githubusercontent.com/53552847/129351646-31b8d410-e6fa-4408-95cb-39dac7e0a352.PNG)
- 위의 수식은 Discriminator 수식이다.
- 더불어, Discriminator를 최적화하는 D는 다음과 같다. 이러한 D를 Optimal Discriminator라고 한다.
![DL Basic 53](https://user-images.githubusercontent.com/53552847/129351649-3b07a426-e0b2-4f4e-bd4d-b14f07cc920c.PNG)
- 이런 Optimal Discriminator를 다시 Generator에 집어 넣어주게 되면 다음과 같은 식이 된다.
![DL Basic 54](https://user-images.githubusercontent.com/53552847/129351651-244ea741-9aa1-4b56-96e9-470aec04dd20.PNG)
- GAN의 Object가 많은 경우에 우리의 True Data Degenerative Distribution, 즉 우리가 데이터를 실제로 만들어 냈다고 생각하는 뭔지 모르는 Distribution과 내가 학습한 Degenerator 사이에 Jenseon-Shannon Divergence를 최소화하는 것이다.라고 많이 얘기한다.
- 실제로 봤을 때, Discriminator가 Optimal Discriminator로 수렴하는 것을 보장하기 힘들고, 이렇게 됬을 경우 Generator가 위의 식처럼 잘 안나올 수도 있기 때문에  이론적으로는 당연히 말이 되지만, 현실적으로 Jenson-Shannon Divergence를 줄이다는 것은 약간 의아한 점이 있다.
- 위에서 말한 논리가 앞에서 봤던 Adversarial Auto Encoder를 와설스텐 Auto Encoder(수정해야함)로 해석할 때 활용한다.
- GAN Objective는 나의 True Generative Distribution과 내가 학습하고자 하는 Generator 사이에 Jenson Shannon Divergence를 최소화한 거라고 말한다면 위에서 나타난 식을 의미하는 것이다.

### 2-4. DCGAN
- 일반적인 GAN은 MLP, 즉 Dense Layer로 만들었다.
- 이것을 Image Domain으로 했다라는 것이 DCGAN이다.
- 당연히, Generative Deconvolution Error를 활용했으며, Discriminator Deconvolution Network를 활용했다.
- 이미지를 만들 떄는, MLP를 사용하는 것보다는 Deconvolution Layer를 활용해서 Generator를 만드는 것이 더 성능이 좋다.

### 2-5. Info-GAN
![DL Basic 55](https://user-images.githubusercontent.com/53552847/129351652-c6a4a21b-3564-48ab-8166-fd5fe9c35201.PNG)
- 위의 그림처럼 Class C라는 Auxiliary Class를 매번 랜덤하게 집어넣는다. 이렇게 하면 Class C를 랜덤한 원핫벡터로 볼 수 있다.
- 결과론적으로 Generation을 할 때, GAN이 특정 모두를 집중할 수 있게 만들어 준다. 그 특정 모두라는 것은 C로부터 나오는 원핫벡터이고, Conditional Vector에 집중할 수 있게 한다. 마치, Multimodal Distribution을 학습하는 것을 C라는 벡터를 통해서 약간 잡아주게 된다.

### 2-6. Text2Image
- 문장이 주어지면 이미지를 만든다.

### 2-7. Puzzle GAN
- Image 안에 subpatch가 있다.
- Subpatch가 있을 때 원래 이미지를 복원하는 Network를 GAN을 통해서 만든다.

### 2-8. Cycle GAN
- GAN구조를 사용하는 데 이미지 사이의 도메인을 바꿔준다.
- Cycle-Consistency Loss를 활용한다. -> 얘는 굉장히 중요하다. 꼭 알아두어야 할 Concept이다.
- Cycle-Consistency Loss의 가장 큰 장점
    - 원래는 두 개의 도메인 사이에 두 개의 똑같은 이미지에 서로 다른 도메인이 있는(얼룩말과 그냥 말) 사진이 필요한데 여기서는 필요없다. 그냥 야생을 떠도는 말 사진과 얼룩말 사진이 잔득있었을 때, 알아서 임의의 말이 있는 사진이 주어지면 얼룩말 이미지로 바꿔준다.

### 2-9. Star-GAN
- 이미지를 다른 도메인으로 바꾸는 것이 아니라 컨트롤 할 수 있도록 한다.

## 3. 피어 세션
### 3-1. [팀 회고록 정리]

각자 회고록을 작성할수 있는 시간 조금 가지기.
각자의 회고를 간단하게 발표하기.

### 3-2. [스페셜 피어세션]
다른 팀들은 알고리즘도 많이 푼다.
우리도 논문을 읽어보자 !! -> 원래 매주 금요일마다 2명이 발표하던것을 빼고 매주 논문을 읽고 목요일마다  이야기 하자
이번 논문은 “Attention is  all you need”로 결정. 
 
### 3-3. [일정변경]
원래 오늘 있을 2명의 마지막 발표를 다음주 화요일로 옮기자.

## 4. 10일차 후기
금일 Generator에 대한 내용을 학습했다. 이전 내용과는 차원이 다른 Advanced한 수준임을 느꼈고 본격적으로 쉬는 시간은 없어질 것임을 직감할 수 있었다. 1기 후기를 보면서 들었던 이야기가 정말 사실이었구나를 느낄 수 있었던 한 주였던 것 같다. 주말동안 많이 보충해야 겠다 ㅜㅜ

금일, 스페셜 피어세션을 통해 다른 조의 팀원들과의 회의를 통해서 서로 어떻게 피어세션 시간을 보내는지에 대하여 이야기하고, 서로의 의견을 공유해보기도 하였다. 사실, 스페셜 피어세션에 대한 어떤 계획도 정해지지 않은 채로 임하다보니 막막했는데, 다른 팀원들이 말을 잘 이끌어주셔서 이야기를 잘 나눌 수 있었고 좋은 정보들도 공유할 수 있는 시간이 되었던 것 같다!

마지막으로 마스터클래스를 통해 교수님께서 강의를 진행해주셨는데, 현재의 나로서는 따라가는게 정말 힘들었다. 아직 나의 수준은 바닥 밑임을 깨달았고 이상하지만 더 열심히 해야겠다는 생각을 많이 들게 했던 것 같다. 더불어, 이런 강의를 잘 모르지만, 들음으로서 이후에 공부를 진행하는 중에 문득문득 생각나지 않을까라는 생각을 하며 꾹 참고 들었다. 아, 꾹 참고 들었긴 한데 사실 조금은 졸았던 것 같다. 하하.

그래도 이번 한주를 나름 만족하게 마무리했던 것 같고, 주말은 보충으로 전부 보내야할 것 같다. 롱런을 하려면 휴식과 공부를 잘 배분해야한다고 하는데 아직은 그런 노하우가 부족한 것 같다. 너무 지치지 않을 정도로 공부했던 내용을 복습 및 보충을 할 수 있는 주말이 되면 좋겠다.

## 5. 해야할 일
- Generative Model의 정의
- Discriminative Model의 정의
- Possible State는 $$2^n$$으로 동일하지만 사용되는 Parameter의 수가 달라진다. -> Possible State란 무엇인가?
- Pixel RNN?? 보충
- Posterior Distribution이란?
- [수정]이라고 써놓은 곳 수정하기
- Latent Vector가 무엇인가?
- Latent space가 무엇인가?
- Latent Distribution 사이의 워스슈타인 디스턴스를 줄여주는 것과 똑같은 효과가 있다.(무슨 말이지..?)
- Cycle-Consistency Loss란?

