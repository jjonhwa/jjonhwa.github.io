---
layout: post
title: "Ustage Day 8"
categories: boostcamp
tags: main
comments: true
---
CNN에 대한 설명을 진행한다.

**부스트 캠프 8일차 학습 요약**
- **행사** : Git 특강, 멘토링
- **학습** : Convolution은 무엇인가?, Modern CNN, Computer Vision Applications
- **피어 세션**

## 목차
- [1. Convolution은 무엇인가?](#1-convolution은-무엇인가)
- [2. Modern CNN](#2-modern-cnn)
- [3. Semantic Segmentation](#3-semantic-segmentation)
- [4. 피어 세션](#4-피어-세션)
- [5. 8일차 후기](#5-8일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Convolution은 무엇인가?
### 1-1. Convolution
- 두 개의 함수를 잘 합쳐주는 혹은 Operator를 의미한다.
![DL Basic 15번](https://user-images.githubusercontent.com/53552847/129023693-e378a5de-99a7-4b0a-a809-d4c693dd949a.PNG)

### 1-2. 2D Convolution의 의미
- Blur, Emboss, Outline 등의 의미가 될 수 있다.
- Filter의 Channel과 Image의 Channel은 같다.
- 결과의 Channel이 1이 아니라면 결과의 Channel 수 만큼의 Filter가 활용된 것이다.
![DL Basic 16번](https://user-images.githubusercontent.com/53552847/129023695-118d02ce-3ddd-424c-8f5c-46851f0ba815.PNG)

### 1-3. Convolutional Neural Network의 구성
- Convolution Layer, Pooling Layer, Fully Connected Layer로 구성되어 있다.
- Convolution & Pooling Layer : Feature Extraction (정보 추출)
- Fully Connected Layer : Decision Making (분류, 회귀 등의 우리가 원하는 출력을 얻게 해준다.)
- 내가 학습시켜야 하는 Parameter의 숫자가 늘어나면 늘어날수록 학습이 어렵고 Generalization Performance가 떨어진다는 것으로 알려져 있다.
- Layer 별로 몇 개의 Parameter로 이루어져 있고, 전체 Parameter의 숫자가 몇 개인지에 대하여 항상 잘 파악하고 있는 것이 중요하다.

### 1-4. Concept of Notation
- Stride
    - Kernel을 몇 칸 마다 적용할 것인가.
- Padding
    - 기존의 Input의 테두리에 Padding을 해줘서 가장자리에 대한 정보를 유지할 수 있도록 한다.
![DL Basic 17번](https://user-images.githubusercontent.com/53552847/129023696-fbd6b770-2b73-4cb9-ac84-9ba664664fcb.PNG)

-  Parameter
    - Kernel size * Input Channel * Output Channel
    - Stride, Padding은 Parameter 수와는 무관하다.
    - Convolution Network에서는 Kernel이 모두 동일하게 적용되기 떄문에 Parameter의 수가 Fully Connected Layer에 비해 적다.
    - Fully Connected Layer를 줄이고 Convolution Network를 깊게 쌓음으로서 전체 Parameter의 수를 줄이는 것이 트렌드이다.

### 1-5. 1x1 Convolution
- Dimension Reduction (Channel의 수를 줄일 수 있다.)
- Convolution Network를 깊게 쌓으면서 동시에 Parameter 수를 줄일 수 있다.
- Bottleneck Architecture

## 2. Modern CNN
- Network의 Depth는 깊어가고 Parameter의 수는 점점 줄어드는 추세이다.
- 5개의 Network들의 주요 아이디어와 구조에 대하여 배운다.
- AlexNet
- VGGNet
- GooLeNet
- ResNet
- DenseNet

### 2-1. AlexNet
- 11x11 Kernel을 활용한다. But, Parameter 입장에서 11x11을 활용하는 것은 좋은 선택이 아니다. -> VGG를 통해 알 수 있다.
- 11x11 Kernel을 활용하면 볼 수 있는 영역이 넓어지지만 Parameter의 수가 많이 커진다.
- 5개의 Convolutional Layers, 3개의 Dense Layers로 구성되어 있다.
- Key Ideas
    - ReLU Activation 활용
        - Linear Model들이 가지고 있는 좋은 성질들을 가지고 있다.
        - 최적화하기 쉽다.
        - Generalization이 좋다.
        - Gradient Vanishing 문제를 극복한다.
    - 2개의 GPU 활용
    - Overlapping Pooling
    - Data Augmentation
    - Dropout

### 2-2. VGGNet
- 3x3 Kernel만을 활용하였다.
    - 3x3 Kernel을 두 번 활용하는 것과 5x5를 한 번 활용하는 것은 Receptive Field가 같다. 즉, 층은 깊어지면서 Parameter의 수가 동일하므로 더 좋은 효과를 거둘 수 있다.
    - 요즘 트렌드는 7x7 Kernel을 벗어나지 않는다.
- 1x1 Convolution for Fully Connected Layer(중요한 것은 아니다.)
- Dropout(p=0.5)
- Layer의 개수에 따라 VGG16, VGG19라고 불린다.

### 2-3. GoogLeNet
![DL Basic 18번](https://user-images.githubusercontent.com/53552847/129023697-46c94203-5a4b-4ab2-8ee6-502da55a547b.PNG)
- 1x1 Convolution Net을 중간중간에 잘 활용하여, 어떻게 하면 전체적인 Parameter의 수를 줄일 수 있을 지에 대하여 알 수 있다.
- Inception Block을 활용한 Network-in-Network
- Inception Block은 다음과 같다.
![DL Basic 19번](https://user-images.githubusercontent.com/53552847/129023700-35f46d52-cb37-4758-a656-f26d771c28ad.PNG)
- Inception Block을 활용함으로서 Parameter의 수를 줄일 수 있다.
- 1x1 Convolution은 Channel-wise 차원 축소를 해준다.

### 2-4. ResNet
- Parameter의 수가 많으면 Overfitting이 일어날 수 있다.
- Network가 깊어진다고 해서 학습을 더 잘 시키는 것만은 아니다.
- Residual Connection(Identity Map)을 추가하게 된다.
![DL Basic 20번](https://user-images.githubusercontent.com/53552847/129023703-18324107-b0c8-4ae5-bcfc-773e5bf1c547.PNG)
- ResNet을 활용하게 깊게 쌓을수록 학습이 더 잘 된다.
- Short Cut의 값을 더하기 위해서는 차원이 같아야한다. 이 때, 차원을 맞춰주기 위해서 1x1 Convolution을 활용하는데, 요즘에는 많이 사용하지는 않는다.
- Batch Normalization이 Convolution 뒤에 일어난다.(논란이 많다.)
- Bottleneck Architecture
    - Conv 앞단과 뒷단에 1x1 Conv를 활용한다.
    - 이렇게 해서 궁극적으로 우리가 원하는 Output Dimension을 만들 수 있다.

### 2-5. DenseNet
- ResNet이 원래 것과 ShortCut을 + 했다면, DenseNet은 원래 것과 ShortCut을 Concatenation([,]) 한 것이다.
- Channel이 커짐에 따라 Convolution Feature Map의 크기 역시 기하급수적으로 커진다.
- 중간 중간에 Convolution을 활용하여 Parameter의 수를 줄여준다.
![DL Basic 21번](https://user-images.githubusercontent.com/53552847/129023704-21d2c2b1-bf44-49f6-96ac-caa3653fd6d6.PNG)
- Dense Block으로 키운 후에 1x1 Conv를 활용해 줄여주고를 반복한다.
- Dense Block
    - 각 Layer를 지나는 모든 레이어에 Feature Map을 Concate한다.
    - Channel의 수가 기하급수적으로 커진다.
- Transition Block
    - Batchnorm -> 1x1 Conv -> 2x2 AvgPooling
    - 차원 축소
- Network를 만들어서 무언가를 해야겠다고 한다면, 보통 ResNet 혹은 DenseNet 구조를 활용한다면 웬만큼의 성능은 잘 나온다.

### 2-6. Summary
- VGG : Repeated 3x3 Block
- GoogLeNet : 1x1 Convolution
- ResNet : Skip-Connection
- DenseNet : Concatenation

## 3. Semantic Segmentation
- 자율주행에 사용된다.
- 이미지로 해결하는 문제에서는 Semantic Segmentation을 잘 활용하는 것이 매우매우 중요하다.

### 3-1. Fully Convolutional Network
- 'Dense Layer를 없애고 싶다.'에서 출발하였으며, 이를 활용하여 Dense Layer를 없앨 수 있다.
- Dense Layer를 활용하나, Fully Convolutional Network를 사용하나 Input과 Output은 동일하다. Parameter가 정확히 일치한다.
![DL Basic 22번](https://user-images.githubusercontent.com/53552847/129023705-419c9e07-197f-42a9-9b2e-bbc6b5e6f4b0.PNG)

- Prameter, Network, Output 전부 똑같은데 Semantic Segmentation 관점에서 왜 Dense Layer를 사용하지 않고 Fully Convolutional Network를 사용하는가?
- Convolution의 경우 Input image의 크기와 상관없이 Kernel이 동일하게 찍기 때문에, Resulting Special Dimension만 같이 커지지 여전히 동작시킬 수 있다.(Dense의 경우 이미지가 너무 커지게되면 동작시킬 수 없을 수 있다.)
- Convolutionalization을 하게 되면, 이미지가 커짐에 따라 분류만 했던 것도 Semantic Segmentation을 할 수 있게 된다.
- Fully Convolutional Network(FCN)은 어떠한 input size에 대해서도 작동할 수 있다.
- Output Dimension은 줄어든다. (Coarse Output을 Dense Pixel로 바꿔줘야한다. -> 늘리는 방법이 필요하다.)

### 3-2. Deconvolution(Conv Transpose)
- 쉽게 이야기하자면 Convolution의 역연산(하지만 실제로 Convolution의 역연산을 구할수는 없다.)
- Dimension을 키워주게 된다.

### 3-3. Detection
- Bounding Box를 찾는 것이다.

#### R-CNN
- Extract Region Proposals (크기에 상관없이 일단 Bounding Box를 뽑는다.)
- Compute CNN Features (Bounding Box를 똑같은 크기로 맞춰서 CNN을 실행한다.)
- Classify Region (각 Boudning Box에 대하여 라벨을 추출한다.)

#### SPPNet
- R-CNN의 문제 -> 이미지 안에서 Bounding Box를 N개를 뽑으면 N개의 이미지 혹은 팻치를 전부 CNN에 통과시켜야 한다. 즉, N개가 모두 Conv Net을 통과해야지 하나의 이미지가 돌어가게 된다.
- 한 이미지에서 CNN을 한 번만 통과하면 추출할 수 있도록 하자.
- Image에서 Bounding Box를 뽑고 이미지 전체에 대해서 Convolution Feature Map을 만든 후, Bounding Box에 위치하는 Convolution Feature Map의 Tensor를 긁어온다. 이 Tensor를 뽑아오는 것만으로도 Region 별로 진행하기 때문에 R-CNN에 비해 훨씬 빠르다.

#### Fast R-CNN
- SPPNet과 비슷한 아이디어 활용
- Bounding Box를 미리 몇 개 꺼내온다.
- Convolutional Feature Map을 한 번 얻는다.
- 각 Region에 대하여 Fixed Length Feature를 ROI Pooling을 통해 뽑느다.
- Neural Network를 통해서 내가 얻은 Bounding Box를 어떻게 움직이면 좋을지를, 그리고 Label을 찾게 된다.

#### Faster R-CNN
- Image를 통해서 Bounding Box를 뽑아내는 Region Proposal 역시 학습하자.
- Region Proposal Network + Fast R-CNN
- Region Proposal Network(RPN)
    - 이미지가 있을 때, 이 이미지의 특정 영역(Patch)이 Bounding Box로서의 의미가 있을 지 없을지를 찾아준다. 어떤 의미인지는 모른다. 
![DL Basic 23](https://user-images.githubusercontent.com/53552847/129023706-8f95edad-50ae-4a28-ad87-fdf2a67faf0a.PNG)
- Anchor Box : 미리 정해놓은 Bounding Box의 크기 -> 대충 이러한 크기의 Object가 이 이미지 속에 들어있다는 것을 알고 있어야 한다.
- K개의 템플릿들을 미리 만들어 놓고, 얼마나 바뀔지를 찾고 궁극적으로 템플릿을 미리 고정해놓는 것이 RPN의 가장 큰 특징이다.
- Fully Convolution Network가 사용된다.

#### YOLO
- 어느 정도 잘 작동하며, 앞서 등장한 모델들 보다 훨씬 빠르다.
- Resion Proposal에 대한 Step이 없기 떄문에 한 번에 한다. (Object의 위치와 그게 무엇이 의미하는지를 한 번에 알아낸다.)
- Anchor Box가 없다.
- 과정
    - 이미지가 들어오게 되면 이를 SxS Grid로 나눈다.
    - 이 이미지 안에, 우리가 찾고 싶은 물체의 중앙이 있고, 그 중앙이 Grid 안에 들어가면 그 Grid Cell이 해당 물체에 대한 Bounding Box와 그 해당 물체가 무엇인지를 같이 예측해준다. 
    - 각 Cell은 B개의 Bounding Box를 예측하게 된다.
    - 각 Bounding Box의 x,y,w,h를 찾아주고 그 Bounding Box가 실제로 쓸모있는지 없는지(Box Probability)를 찾아준다. 
    - 그와 동시에, 각각의 SxS Grid가 이 Grid Cell에 속하는 중점에 있는 Object가 어떤 Class인지를 예측하게 된다. 
    - SXSX(B*5+C)






## 4. 피어 세션
### 4-1. [이전 질문 리뷰]

ViT(Visual Transformer): 금일(21210811) 19시 멘토링 시간에 질문하기로 함.
각 Patch를 차원이 줄여진 잠재 벡터(latent vector)로 볼 수 있는가?
MNIST dataset 사용 기준 각 (4,4)의 크기의 Patch 49개가 생성된다. 

### 4-2. [금일 질문 목록]:
CNN 수강 관련 질문
Dense layer를 잘 이해하지 못하겠다.
Fully Connected Layer과 같은 것.
Dense layer의 가중치 또한 학습이 진행되는가?
에러에 따라 모든 (Conv, FCN)의 가중치들의 학습이 진행된다.
Convolution 연산에서 가중치의 차원은 (kernel_W, kernel_H, in_channel, out_channel)과 같다.  그렇다면 in_channel 기준으로 같은 값으로 연산이 진행되는지(Broadcasting) 또는 각 in_channel마다 다른 값을 가지는지 궁금하다.
강의에서 학습할 가중치의 개수가 “kernel_W * kernel_H * in_channel * out_channel” 이라고 설명한 것으로 미루어 보아, 각각 다른 in_channel 가중치는 각각 다른 값을 가질 것이라고 생각된다.
e.g. in_channel 기준 각각 다른 가중치의 값은 다음과 같다. (parameter[0,0,:,0])

### 4-3. [선택과제 3번 살펴보기]
Mixture_density_Network_문제
일대일 대응이 아닌 그래프에 대해 회귀(근사)를 어떻게 할 것인가?
위 식에서 변수 y의 의미
가우시안 분포의 변수로 생각됨.
또한 위의 식은 x, y축이 서로 바뀐 것으로 생각됨.
과제 설명 중 MSELoss(Mean Squared Error Loss)를 사용하지 못한다고 했다. 이에 관한 질문.
로그 우도(Log Likelihood)를 사용하려고 한 것 같다.
 Gumbel softmax sampling에 대해 알고싶다.

### 4-4. [ViT 관련 추가 질문]
Q, K, V 가 각각 의미하는 것이 무엇일까?
단어와의 연관성, cosine 유사도 관련 설명 진행함


## 5. 8일차 후기
Git 특강을 통해서 Git에 대한 전반적인 활용도에 대하여 알게 되었고, 이를 활용한 협업 방법을 배울 수 있었다. 그동안 혼자서 공부하고 혼자서 정리하는 습관이 들여져 있었지만 함께하는 것에 서툴었기 때문에 정말 정말 소중한 강의였고 이를 바탕으로 Git에 조금 더 익숙해지고 협업에도 입문을 해보고 싶다.

더불어, CNN에 대한 내용을 배움으로서 CV에 대한 흥미가 조금은 더 올라갔던 것 같다. 사실, 이전에도 CNN에 대한 내용을 공부했지만 벽에 부딪힌 감이 있었는데 이번 강의를 통해서 어느 정도 이해할 수 있었고 조금 더 보충을 한다면 어느 정도의 감을 가지고 앞으로의 강의 및 프로젝트에 임할 수 있을 것이라고 생각한다.

## 6. 해야할 일
- Dense Layer가 Mulilayer Perceptron과 똑같은가?
- 꼭 Dense Layer를 사용해야하는가?
- Overlapping Pooling?
- Receptive Field란?
- ResNet은 기존의 정보를 함께 활용하는 기법인건가?
- Further Question : 수업에서 다룬 Modern CNN Network의 일부는, Pytorch 라이브러리 내에서 Pre-trained 모델로 지원한다. Pytorch를 통해 어떻게 불러올 수 있을까?(https://pytorch.org/vision/stable/models.html 참고)
-  Dense Layer를 활용하나, Fully Convolutional Network를 사용하나 Input과 Output은 동일하다. Parameter가 정확히 일치한다. -> 완전히 똑같은가?
-  Fully Convolutional Network에 대한 보충
-  Fast R-CNN : '각 Region에 대하여 Fixed Length Feature를 ROI Pooling을 통해 뽑느다.'은 무슨말?
