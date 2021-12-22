---
layout: post
title: "Variational AutoEncoder"
categories: booststudy
tags: plus
comments: true
---
Variational AutoEncoder에 대하여 학습한다. 본 글은 [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)를 참고하여 작성하였습니다.

## 목차
- [1. AutoEncoder](#1-autoencoder)
- [2. Variational AutoEncoder](#2-variational-autoencoder)

## 1. AutoEncoder
- AutoEnocder는 Encoder와 Decoder를 연결한 Network이다.
- Encoder: data를 압축하여 Dense Representation으로 표현한다.
- Decoder: 압축된 Dense Representation을 Original Input으로 복원한다.

### 1-1. Encoder and Decoder
#### Encoder
- Data를 압축하여 Dense Representation으로 표현한다.
- 그렇다면, Dense Representation으로 표현한다는 것이 무엇을 뜻하는 것일까?
    - CNN을 예로 들면 다음과 같다.
    - convolution network를 활용하여 224x224x3 RGB image를 64x64x32 -> 32x32x128 -> 1x1x1000과 같이 flatten한 vector로 만드는 과정과 유사하다고 보면 이해하기 쉽다.
    - 이 때, flatten된 vector representation을 Compress된 Dense Representation이라고 이해할 수 있다.
- Encoder는 Back-Propagation 과정에서 함께 학습된다.
- Encoder의 output은 input보다 훨씬 적은 단위를 가지므로, 어떤 정보를 버리고 어떤 정보를 얻을 것인지 선택해야한다. 즉, 이 과정을 전체적인 AutoEncoder에서 Encoder가 학습하게된다.

#### Deocder
- Encoder로부터 나온 Dense Representation을 바탕으로 원본 이미지를 잘 복원할 수 있도록 학습한다.

### 1-2. Problem with standard autoencoders
- AutoEncoder는 입력을 압축하고 이를 다시 재구성하도록 학습하는데, 이 때 Denoising AutoEncoder와 같은 몇 가지 Application에서는 상당히 제한적이다.
- 근본적인 문제는 Latent Space가 연속적이지 않거나 Easy Interpolation을 허용하지 않는다는 것이다.
- 예를 들어, MNIST로 AutoEncoder를 학습시킬 때, 2D Latent Space를 시각화하면 아래 그림과 같이 뚜렷한 cluster로 나뉘어지는 것을 볼 수 있다.
![1](https://user-images.githubusercontent.com/53552847/147046582-6b452307-ab4e-4c34-8bb5-b8423a8d1368.jpg)
- 위와 같은 latent space에서의 뚜렷한 군집으로서부터 decoder가 다시 복원해내기 더욱 쉽게 만들지만, 이것은 복제할 때 좋은 것이지 보통 input image의 variation을 생성하고 싶거나 랜덤하게 생성하고 싶을 것이다.
- 이러한 문제로부터, Decoder가 cluster 되지 않은 latent space에서 복원해 내는 방법을 학습하지 못하기 때문에 real하지 않은 이미지를 출력할 것이다.
- 이러한 문제로부터 Variational AutoEncoder가 연구되기 시작했다.

## 2. Variational AutoEncoder
- Standard AutoEncoder와는 다른 고유 속성이 있는데 이러한 속성으로부터 latent space의 설계를 연속적으로 만들고, 이를 바탕으로 random sampling and interpolation을 용이하게 만들어준다.
- Standard AE에서는 Latent Representation을 size가 n인 Encoding Vector로 표현했다면, VAE에서는 평균 벡터 $$\mu$$와 표준편차 벡터 $$\sigma$$를 출력한다.
- n개의 random variables인 $$X_i$$를 생성하는데, 이 $$X_i$$는 $$i_{th}$$ 평균과 표준편차를 가지는 Normal Distribution에서의 Random Variable이 된다.
- 이로부터, Decoding될 Sample Encoding Vector를 얻게된다.
- 이를 바탕으로한 Stochastic한 Generation은 동일한 입력에 대해서도 서로 다른 Embedding Vector를 출력해낼 수 있다는 것을 의미한다.
![2](https://user-images.githubusercontent.com/53552847/147046588-857fa9b2-ba01-429b-ac4c-7b3142f188b7.jpg)

### 2-1. KL Divergence Loss
- Reconstruction은 Dense Representation으로부터 복원을 하는 과정이고, 우리들이 이상적으로 원하는 것은 Dense Representation을 구별되도록 샘플링하는 것이다.
- 위의 과정에서 Dense Represenation이 서로 구별되면서 새로운 샘플링을 할 수 있도록 하였다.
- 이와 더불어, VAE에서는 이러한 distinct sample and interpolation을 강화하기 위하여 KL Divergence Loss를 활용하였다.

#### KL Divergence
- 두 확률 분포 사이의 KL Divergence는 단순히 얼마나 차이가 있는지 측정한다.
- KL Divergence를 최소화한다는 것은 확률분포의 모수인 $$\mu$$와 $$\sigma$$를 우리가 target으로 하는 distribution과 비슷하게 최적화하는 것을 의미한다.
- VAE의 경우, KL Loss가 모든 $$X_i$$들 사이의 KL Divergence의 합과 같으며, $$\mu$$ = 0, $$\sigma$$ = 1일 경우 최소화된다.
- 직관적으로, 이러한 KL Loss로부터, 중심에서 벗어나려고 하는 encoding vector에 대하여 패널티를 주기 떄문에 모든 encoding vector의 분포가 latent space의 중심에 분포할 수 있도록 한다. 
![3](https://user-images.githubusercontent.com/53552847/147046591-908e5ec9-d042-4111-85ce-e63a81f21014.jpg)
- 하지만, 이러한 분포를 활용해서는 Decoder가 의미있는 것을 decoding하기에는 어렵게된다.

#### Optimize Encoder with Decoder
- 위의 문제로부터, Encoder와 Decoder를 함께 최적화 시킨다.
- Encoder와 Decoder를 함께 최적화시키게 되면, Clustering을 통해 Encoding vector가 각 군집들끼리 유사한 Latent space를 생성하며 이와 함께, 원점에 밀집될 수 있도록 최적화된다.
![4](https://user-images.githubusercontent.com/53552847/147046592-d950dd3b-c463-4ecf-ac8d-d0612020cd9c.jpg)
- 이렇게 함으로서, Reconstruction Loss를 바탕으로한 latent space의 군집 형성과, KL Loss를 바탕으로한 중심 밀집 분포를 통해서 Decoder가 이해할 수 있는 Dense Representation을 형성할 수 있게 된다.

### 2-2. Create a new image with different characteristics.
- Sample Distribution의 차이를 활용하여, 다른 특징을 가지는 새로운 이미지를 생성할 수 있다.
- 예를 들어, '얼굴 -> 안경 쓴 얼굴'과 같은 이미지를 생성할 수 있다.
- latent space에서의 vector의 움직임은 다음과 같은데, 어떤 두 샘플 사이의 새로운 샘플을 만들거나 특징에 대한 vector를 찾아서 저장하고 이를 활용할 수도 있다.
![6](https://user-images.githubusercontent.com/53552847/147046789-62f8a900-ac38-43af-a9d2-a1cbdf32f6a6.jpg)
![7](https://user-images.githubusercontent.com/53552847/147046791-81bb6259-7517-486b-9da9-074776f251bf.jpg)
- 즉, 두 샘플 사이의 새로운 샘플을 만들 때는, 두 샘플에 대한 $$\mu$$간의 차이를 계산하여 이의 절반을 원본에 추가한 후 decoding하면 두 샘플 사이의 새로운 샘플을 만들 수 있다.
- 더불어, 앞서 얘기한 것과 같이 "얼굴 -> 안경 쓴 얼굴"을 만들고 싶은 경우, 안경을 쓰지 않은 얼굴과, 안경 쓴 얼굴 사이의 encoding vector간의 차이를 저장한 후 이를 안경 쓰지 않은 얼굴에 추가하여 decoding하면 안경 쓴 얼굴이 생성될 수 있다.
