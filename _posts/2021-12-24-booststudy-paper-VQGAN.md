---
layout: post
title: "[논문리뷰] VQGAN: Taming Transformers for High-Resolution Image Synthesis"
categories: booststudy
tags: paper
comments: true
---
VQGAN: [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)를 읽고 이에 대하여 논의한다.

## 목차
- [1. Abstract](#1-abstract)
- [2. Introduction](#2-introduction)
- [3. Related Work](#3-related-work)
- [4. Approach](#4-approach)
- [5. Experiments](#5-experiments)
- [6. Conclusion](#6-conclusion)
- [7. 해야할 일](#7-해야할-일)


## 1. Abstract
- **Transformers는 CNN에 반해 Local Interaction을 일으키는 Inductive Bias를 포함하지 않는다.**
    - 즉, 표현력이 강하고, **High-Resolution**을 나타낼 수 있다.
    - 하지만, High-Resolution을 연산할 때, **Computational Cost**가 엄청나기 때문에 Resolution을 높이기에 무리가 있다.
- **CNN은 Inductive Bias를 가지고 있다.**
    - Inductive Bias로 인해서, **Local Interaction**을 잘 이끌어 낼 수 있다.
    - 더불어, **학습을 할 때 좀 더 효율적**이다. (Kernel에서의 Parameter Sharing)
    - 하지만, Image의 전체적인 특성을 잡아내는 데에는 무리가 있다. 즉, **표현력에서는 Transformer에 비해 떨어진다.**
- 각각의 장단점을 가지고 있는, **Transformer와 CNN에서 장점만을 취해서 활용한다면 High-Resolution Image를 생성**해 낼 수 있지 않을까?!
    - **CNN을 활용하여 Local에 집중한 상태로 Codebook을 만든다.**
    - **Codebook을 바탕으로 Transformer를 활용하여 Computational cost를 낮추는 것에 더불어 Global feature를 더 잘 잡아낼 수 있다.**
- 더불어, Conditional Synthesis task에 잘 적용된다. 

## 2. Introduction
- CNN의 경우, Image 안에서 Strong Local Correlation을 가지고 있고 이에 대하여 Design 해낼 수 있다.
- 하지만, Strong Locality Bias에 더해, 모든 position에서의 weight share로부터 Bias toward sptial invariance를 나타냄으로 인하여 image 전체를 globally understanding하기가 어려워진다.
- Transformer을 활용하면 좀 더 Global하게 이미지를 이해할 수 있다. 하지만, local connectivity를 잘 설명하지 못하고, 이에 대하여 본 논문에서는 더 높은 의미론적 수준에서는 transformer가 더 이상 효과적이지 않다라고 가정한다.
- **핵심 insight로서 CNN과 Transformer를 함께 사용해서 효과적이고 더 표현적인 모델을 만들고자 한다.**
- 먼저, **CNN을 활용하여 풍부한 Codebook을 학습시킨 이후 Transformer를 활용해 Global Composition을 학습한다.**
- transformer를 활용해 long-range interaction을 표현하고, high-resolution image를 생성할 수 있도록 한다.
- 기존에 존재하는 CNN, codebook 기반 approach에 대비하여 Transformer의 장점을 유지하여 성능이 더 좋아진다라는 것을 입증한다.

## 3. Related Work
- 다양한 Related work이 있지만, 그 중에서도 가장 관련 깊은 **VQ-VAE**에 대해서만 간략하게 소개하고 넘어가도록 한다.
- **VQVAE에서는 Image를 Discrete Representation으로 표현하여 학습하고, 이를 CNN과 함께 Distribution Auto-Regressive한 모델링**을 진행하며, 이를 바탕으로 **Hierarchy of learned Representation의 접근**을 확장한다.
- 하지만, 이런 방법론은 CNN density estimation에 의존하며, long-range interaction을 통한 high-resolution image 생성에 약점이 있다.
- 32x32만 하더라도 transformer에서는 computationally expensive하지만 VQVAE에서는 transformer를 활용하지 않고도 192x192의 resolution까지 size up 할 수 있었다.
- VQVAE는 small receptive field를 가져가는 반면에 **가능한 한 많은 context를 포착**하는 것이 **transformer**와 함께 **high-resolution image를 합성**하는데 있어 중요하다는 것을 입증한다.

## 4. Approach
전반적인 Approach는 다음의 그림을 활용하며, 각 그림에 대해서는 아래에서 설명하도록 한다.
![1](https://user-images.githubusercontent.com/53552847/147315191-0e049116-0c20-4e6b-b764-aafa37e00c8c.jpg)
### 4-1. Goal
- Transformer를 활용하여 High-Resolution Image Synthesis를 하는 것.
    - Previous work에서 Transformer를 활용한 image synthesis에서 과도한 computational cost 때문에 64x64 image 까지빡에 만들어내지 못하였다. 쉽게 high-resolution으로 scale-up하기가 쉽지 않다.
    - High-Resolution image의 경우 Global composition of image를 이해할 수 있어야하며, 이에 더하여 locally realistic을 생성할 수 있어야 한다.
- **Image를 pixel로 나타내지 않고 Codebook을 활용하여 지각적으로 풍부한 이미지 구성요소를 표현**한다. (CNN을 활용한 풍부한 codebook학습 -> Locally Realistic에 초점)
- **Transformer, Codebook을 활용하여 Global Interrelations을 모델링** (Global composition)
- 위와 같이, Codebook 학습, Transformer 학습을 2 stage로 나뉘어서 진행한다.

### 4-2. Learning an Effective Codebook of Image Constituents for Use in Transformers
- Transformer Architecture를 활용하기 위해서는 이미지의 구성요소를 Pixel로 표현하는 것보다는 Patch 형태(Compressed Image by Encoder)로 표현함으로서 계산량을 줄임과 동시에 이미지의 Global Composition 역시 가져갈 수 있다.
- 더불어, H x W x 3(RGB)의 이미지를 h x w x $$n_z$$ 형태의 Codebook의 code로 표현할 수 있도록 **Discrete Codebook의 학습**을 필요로 한다.
- Discrete Codebook은 VQVAE에서 Idea를 착안했으며 다음의 학습방법에 대한 idea를 제안한다.

#### Idea
![3](https://user-images.githubusercontent.com/53552847/147315189-a061f866-b8da-436c-b2d9-ebb2945e416a.jpg)
- CNN으로 구성된 **encoder E와 decoder G를 학습**한다.
- **x(image) -> (Encoding) -> $$\hat{z}$$ (Compressed Image) -> qunatization(codebook) -> $$z_q$$ (Quantized Image) -> (Decoding) -> $$\hat{x}$$ (Reconstruction Image)**
- **Encoding** (E(x) = $$\hat{z}$$) : 기존 H x W x 3 Image를 압축된 형태인 h x w x $$n_q$$ 로 표현한다.
- **quantization**: $$\hat{z}$$ 과 Codebook에 존재하는 z(각 code) 사이의 거리가 가장 작은 vector를 구한 후, 치환한다. h x w x $$n_z$$ -> h x w x $$n_z$$
![2](https://user-images.githubusercontent.com/53552847/147315190-968fedd9-9a71-4768-b483-52e4537e08d0.jpg)
- **Decoding**: h x w x $$n_z$$ -> H x W x 3
    
#### 학습 과정
- VQLoss + GANLoss (Discriminator Loss)
- VQLoss
![4](https://user-images.githubusercontent.com/53552847/147315188-f862578a-bad1-4c1a-9b41-5ded26ce3b29.jpg)
    - **Reconstruction Loss**: x(original image)와 $$\hat{x}$$ (reconstruction image)가 얼마나 비슷한지 판단한다. 기존(VQVAE)에서는 L2 Loss를 활용하였지만 본 과정에서 Perceptual Loss를 활용한다. (첫번째 항) - Encoder, Deocder 학습
    - **Codebook Loss**: Encoding vector를 stop gradient한 후, Codebook vector가 Encoding vector와 가까워지도록 학습한다. L2 loss 활용(두번째 항) - Codebook 학습
    - **Commitement Loss**: Codebook vector를 stop gradient한 후, Encoding vector가 아무거나 출력하지 않고, Codebook vector와 유사하게 출력할 수 있도록 학습한다. L2 loss 활용 (세번째 항) - Encoder 학습
    - 위 과정을 학습하는 과정에서, backpropagation이 일어날 때, decoder의 backpropagation에서 encoder의 backpropagation으로 이어지는 과정에서 quantization(단순 index mapping)이 일어나는 데, 이 과정에서 어떤 수학적 연관성이 없기 때문에 backpropagation이 일어날 수 없다.
    - 이러한 문제를 해결하기 위하여 gumble softmax 방식 혹은 straight through 방식을 활용할 수 있는데 본 논문에서는 **straight through 방식**을 활용하여 quantization 과정을 skip하고 backprop을 그대로 복사하여 encoder쪽으로 건내주는 방식을 선택하였다. (DALLE의 경우, gumble softmax 방식 선택)
- GANLoss (Learning a Perceptually Rich Codebook)
![11](https://user-images.githubusercontent.com/53552847/147315521-1eccce59-6864-46d6-b4af-42cfae891dad.jpg)
    - image를 latent image constituents에 대한 분포로서 나타내기 위하여 Transformer를 활용하는 것은 Compression의 한계를 극복하기 위해 풍부한 Codebook을 학습해야 한다.
    - **증가된 Compression의 한계를 극복하기 위하여, Discriminator와 perceptual Loss를 활용**한다.
    - 기존 연구에서는 original image와 reconstruction image에 대한 L2 loss를 활용하였지만 본 과정에서는 Reconstruction Loss로서 perceptual loss로 대체한다. 
    - 더불어, real/fake를 전체 이미지가 아닌 **Patch-based Discriminator로 학습**을 진행한다.
    - Discriminator 학습
- Learning an Effective Codebook of Image Constituents for Use in Transformers에서의 전체 loss는 다음과 같다.
![5](https://user-images.githubusercontent.com/53552847/147315186-6333838b-d084-4020-9779-48a5f508e931.jpg)
- 위에서 **lambda의 경우, adaptive weight**을 의미하며, 이를 바탕으로 **VQLoss와 GANLoss 사이에서 어느 하나에 치우치지 않도록 만들어준다.**
    - lambda는 Decoder의 마지막 layer에서의 GANLoss, VQLoss의 변화율을 바탕으로 연산된다.
    - 즉, Gan Loss의 변화율이 커질경우 lambda를 작게 유지하여 최종 loss 계산에서 Gan Loss에 패널티를 주는 방식으로 작동한다. 이로서, VQLoss와 GANLoss의 균형을 유지하도록 한다.
    
### 4-3. Learning the Composition of Images with Transformers
![12](https://user-images.githubusercontent.com/53552847/147315671-6ae580e1-1fa1-46a4-bedb-cc4a6e62a1a4.jpg)
- Latent Transformers
![6](https://user-images.githubusercontent.com/53552847/147315184-63641002-185e-4921-9fd5-65fd3b1f4192.jpg)
    - Trained Encoder E, Decoder G, Codebook Z를 바탕으로 image를 압축된 형태의 quantized image로 표현할 수 있다.
    - 처음에는 random한 값으로 code가 생성되고 이를 바탕으로 transformer가 original image의 quantized image를 label로 활용하여 학습을 진행한다. (image synthesis의 경우 random code가 아닌 image를 input으로 받아 predict를 진행한다.)
    - 즉, **transformer를 활용하여 compressed image 형태를 AutoRegressive하게 예측**하게 된다. 이후 과정에서 이를 decoder를 통과시켜 reconstruction 시켜준다.
    - transformer의 경우 **log-likelihood를 최대화하도록 학습**을 진행한다.
- Conditioned Synthesis
![7](https://user-images.githubusercontent.com/53552847/147315183-d0654e65-1630-4e65-bf43-d903459568b2.jpg)
    -  Condition에 대한 정보를 c라고 했을 때, c가 주어졌을 때, likelihood를 활용하여 auto regressive하게 학습을 진행한다.
    -  (??) **conditioning information c가 spatial extent라면 index-based representation인 r을 다시 얻기 위해, 다른 VQGAN을 학습하고 이로부터 얻어진 codebook $$Z_c$$로부터 quantized 된 vector r을 앞쪽에 condition으로 부착하여 auto-regressive하게 예측**을 진행한다.
    -  이 때, **negative log-likelihood를 활용**하여 계산을 진행한다.
    -  이러한 "Decoder-only" 전략이 text-summarization task에서도 성공적으로 사용되곤 한다.
-  Greating High-Resolution Images
    - **f = H/h로 두고, f는 dataset의 크기에 의존하며, 이 값에 따라서 reconstruction quality가 변화**하는 것을 관찰하였다.
    - megapixel에서 이미지를 생성하려면, 패치별로 작업을 진행할 때, 최대 실현 가능한 사이즈로 패치의 크기를 제한할 필요성이 있고, 이를 f가 조절할 수 있다.
    - 더불어, **image patch를 predict하는 과정에서 sliding window 방식을 활용**하였다.
    ![8](https://user-images.githubusercontent.com/53552847/147315181-02c822c8-1e5a-49d3-a3d6-45d82c95e309.jpg)
    - 하지만, sliding window의 한계로서 landscape에서는 어느 정도 잘 동작하지만, 자전거와 같이 왼쪽, 오른쪽에 균등하게 바퀴가 있어야하는 경우에는 잘 동작하지 않는다고 한다. 즉, spatially dependent할 경우 잘 동작하지 않을 수 있고, sptially independent할 경우 잘 동작한다고 한다.
    - 이러한 방법을 바탕으로 High-Resolution Image를 잘 만들어 낼 수 있다고 한다.

## 5. Experiments
### 5-1. Importance of Transformers
- Shared VQGAN, replace transformer with PixelSNAIL (SOTA, PixelCNN family)
![9](https://user-images.githubusercontent.com/53552847/147315179-5e229ad4-fe66-48e5-9b90-b5456aaa63f0.jpg)
    - Pixel-SNAIL을 활용했을 때, 학습 속도는 더 빠르다.
    - 하지만, 같은 학습 시간 내에서 Transformer의 성능이 더 좋았다.
    - 같은 step 수에서도 Transformer의 성능이 더 좋았다.

### 5-2. Importance of context in the learned codebook
- Amount of context encdoing: f = H/h
    - HxW: input size
    - hxw: discrete code size
    - f가 클 경우, receptive field가 커지기 때문에 transformer는 이미지를 잘 만들어 낸다. 하지만, locally interaction에서는 약점이 있기 때문에, reconstructure를 점점 못하게 된다.
    - 반대로, f가 작을 경우에는, receptive field가 작아지기 때문에 locally interaction에서는 어느 정도 강점이 있어 reconstructure를 해내지만 transformer로 만들어낸 이미지들은 다음과 같이 이상한 모양을 띄게된다.
    ![10](https://user-images.githubusercontent.com/53552847/147315174-c4b9d3a4-a1da-4be0-bc47-52d37da3d368.jpg)
    - 즉, f에 따라서 reconstruction과 transformer 사이의 trade-off가 존재하고 이를 바탕으로 적절한 f를 취해야한다.

## 6. Conclusion
- High-Resolution Image를 Transformer를 활용하여 만들어 낼 수 있다.
- Pixel 단위로 Generation한 것이 아닌, codebook을 활용하여 압축된 형태로 generation을 진행하였다.

## 궁금한 것
- high-resolution image를 가져가기 위하여 풍부한 context를 포착하는 것과 Transformer를 활용하는 것 중에 어떤 것이 진정 영향을 더 많이 끼쳤는가?
- Random함은 어디서부터 오는가?
    - 같은 image를 입력으로 받았을 때, 어디에서 다른 출력으로 이어지는가?!: Transformer를 활용할 때, image를 latent image constituents의 분포로 나타내며 이를 바탕으로 random함이 추가된다.
    - 그렇다면 Transformer가 없을 때는 random함이 오지 않는가?! (ex, VQVAE)
- image sysnthesis 과정에서 새로운 VQGAN을 학습시킬 때, 처음부터 모든 Data를 학습시켜야 하는가? 그렇다면 data는 어떤 것을 활용하는가?!
- 왜, Transformer와 PixcelSNAIL과 비교할 때, 학습을 Optimization하지 않고, 제한된 상황속에서 성능을 비교하였는가?!







