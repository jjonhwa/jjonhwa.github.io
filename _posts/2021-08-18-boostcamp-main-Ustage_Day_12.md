---
layout: post
title: "Ustage Day 12"
categories: boostcamp
tags: main
comments: true
---
Generative Model에 대한 설명을 진행한다.

**부스트 캠프 12일차 학습 요약**
- **행사** : 특별강의(Python Unit Test)
- **학습** : AutoGrad & Optimizer, PyTorch Dataset, 과제
- **피어 세션**

## 목차
- [1. AutoGrad & Optimizer](#1-autograd---optimizer)
- [2. PyTorch Dataset](#2-pytorch-dataset)
- [3. 피어 세션](#3-피어-세션)
- [4. 12일차 후기](#4-12일차-후기)
- [5. 해야할 일](#5-해야할-일)

## 1. AutoGrad & Optimizer
- PyTorch의 Module, Parameter의 Backward 과정 -> Optimizer에 대하여 학습한다.
- nn.Module을 통해 기본적으로 PyTorch에 구현되어 있는 네트워크 텐서 모듈을 살펴보고 nn.Parameter의 Module 내부에서 어떤 역할을 하는지 학습한다.
- Backward 함수가 작동하는 방식을 확인해 직접 구현해보도록 한다.

### 1-1. torch.nn.Module
- 딥러닝을 구성하는 Layer의 base class
- Input, Output, Forward, Backward 정의
- 학습의 대상이 되는 Parameter(tensor) 정의
![pytorch 7](https://user-images.githubusercontent.com/53552847/129924276-e300baa3-fa6a-4490-a8af-351a7683c998.png)

### 1-2. nn.Parameter
- Tensor 객체의 상속 객체
- nn.Module 내에 attribute가 될 때는 'required_grad = True'로 지정되어 학습 대상이 되는 Tensor, 즉, AutoGrad의 대상이 된다.
- 우리가 직접 지정할 일은 거의 없다. 대부분의 layer에는 weights 값들이 지정되어 있다.
- Parameter는 Backward Propagation의 대상이 되는 것들만 저장하기 때문에 Tensor는 저장하지 않는다.(Tensor로 입력하는 것과 Parameter로 입력하는 것의 차이)

### 1-3. Backward
- Layer에 있는 Parameter들의 미분을 수행
- Forward의 결과값(module의 output = 예측치)과 실제값 사이의 차이(loss)에 대해 미분을 수행
- 해당 값을 Parameter를 업데이트
- optimizer.step() : 한 번에 업데이트해야하는 weight 값들이 업데이트 된다.
- 학습이 될 때, 다음 단계는 반드시 거처야한다.
    - optimizer.zero_grad()
    - outputs = model(inputs)
    - loss = loss_function(outputs, labels)
    - loss.backward()
    - otpimzer.step()
**Note :** optmizer.zero_grad()를 하는 이유는, optimzer가 학습될 때, Gradient의 값이 계속해서 Update가 된다. 이 때, 이전의 Gradient 값이 지금의 학습에서 영향을 주지 않도록 하기 위해 초기화를 시켜준다.

### 1-4. Backward from the scratch
- 실제 backward는 Module 단계에서 직접 지정 가능하다. 하지만, AutoGrad가 자동으로 해주기 때문에 직접 지정할 필요가 많이 없다.
- Module에서 backward와 optimizer 오버라이딩 해야한다.
- 사용자가 직접 미분 수식을 써야하는 부담이 있다. 하지만, 쓸일은 많이 없지만 순서는 이해해야할 필요성이 있다.

## 2. PyTorch Dataset
- PyTorch Dataset, DataLoader를 사용하는 방법을 학습한다.
- Dataset 클래스를 학습하여, Image, Video, Text 등에 따른 Custom Data를 PyTorch에 사용할 수 있도록 학습한다.
- DataLoader를 통해 네트워크에 Batch 단위로 데이터를 로딩하는 방법을 배운다.

### 2-1. 모델에 데이터를 먹이는 방법
![pytorch 8](https://user-images.githubusercontent.com/53552847/129924284-1c557e61-edf8-4a6d-a44f-cd8f0a6b43e4.png)
- 전처리된 데이터를 만든다.
- Dataset class를 활용하여, 데이터를 어떻게 불러올 것인지, 총 길이가 얼마인지, 하나의 데이터를 불러올 때 어떻게 반환해주는지(map-style)에 대하여 정의한다.
- transfomrs : 데이터 변형 (전처리와 Tensor로 바꿔주는 것이 구분되어 진행된다.)
- DataLoader : Feeding 시켜주는 역할
    - 보통 batch를 만들어주는 것과 같은 역할을 한다.
    - shuffle을 통해 데이터를 섞는다.

### 2-2. Dataset 클래스
- 데이터 입력 형태를 정의하는 클래스
- 데이터를 입력하는 방식의 표준화
- Image, Text, Audio 등에 따른 다른 입력 정의
![pytorch 9](https://user-images.githubusercontent.com/53552847/129924286-16dd2fc8-2354-4546-837f-0c861218bdc3.png)
- Dataset을 생성할 때, 유의할 점
    - 데이터 형태에 따라 각 함수를 다르게 정의한다.
    - 모든 것을 데이터 생성 시점에 처리할 필요는 없다 -> Image의 tensor 변화는 학습에 필요한 시점에 변환한다.
    - 데이터 셋에 대한 표준화된 처리방법 제공 필요 -> 후속 연구자 또는 동료에게 빛과 같은 존재가 된다.
    - 최근에는 HuggingFace 등 표준화된 라이브러리를 사용한다.

### 2-3. DataLoader 클래스
- Data의 Batch를 생성해주는 클래스
- 학습직전(GPU feed 전) 데이터의 변환을 책임진다.
- Tensor로 변환 + Batch 처리가 메인 업무이다.
- 병렬적인 데이터 전처리 코드의 고민이 필요하다.(요즘에는 많이 줄어들었다고 한다.)
- 하나의 Epic을 돌리려면 DataLoader가 한 번 돌아가면 된다.
- sampler, batch_sampler, collate_fn을 잘 봐두면 좋다.
![pytorch 10](https://user-images.githubusercontent.com/53552847/129924290-9842d564-3c66-4f83-b9c2-6f568299b240.png)
- collate_fn의 경우 Variable을 정의하거나 Padding할 경우에 많이 사용한다.

## 3. 피어 세션
### 3-1. [금일 질문 목록]:
- nn.Linear에서의 크기 변환의 원리는 어떻게 되는가?
    - Linear(N, H) 이므로 H값이 변경
- torch.swipdim(input, dim0, dim1)
- Hook의 실제 사용 사례는 대표적으로 어떤게 있는가?
    - CNN모델에서 filter 설계에 사용되는것으로 알고있다.
- register_forward_pre_hook(), register_forward_hook(), register_full_backward_hook() 의 차이는 무엇인가?
    - forward_pre_hook():
        - forward가 진행되기 전에 어떠한 작용을 하겠다
        - pre_hook : parameter가 어떻게 변하는지 확인할 때 사용, input밖에 없다.
    - full_hook: input output이 있다.
- Apply 문제 해결을 방법은 어떤것이 있는가?
    - apply를 이용하거나 split을 사용하여 해결할 수 있다.
- 일반적으로 Model에서 Parameter정의를 하는데 외부에서 Parameter 정의가 가능한가?
    - 명확한 해답을 결론짓지 못했다.
### 3-2. [금일 과제 분석]
- Dataloader의 num_workers는 무엇인가?
    - 데이터를 불러올때 사용하는 서브 프로세스(subprocess) 개수이다.
- collate_fn의 역할은 무엇인가?
    - [[data1, data2, ...], [label1, label2, ...]] → [[data1,label1],[data2,label2],...]
- batch_size 문제
    - 명확한 해답을 결론짓지 못했다.

## 4. 12일차 후기
PyTorch에 대한 내용을 습득함으로서 아직은 잘 모르겠지만, 무엇인지 모르게 스스로 쑥쑥 성장하고 있는 느낌이 든다. 실습과 과제에 치여서 아니나 다를까 나의 존재에 대한 어떤 작음?을 느끼고 있지만 동시에 어느 정도 성장하고 있는 것도 느낄 수 있어서 나름 뿌듯했다.

과제의 양이 많고 난이도가 높아서 시간이 많이 걸렸지만, 그래도 어느 정도 해결하고 이것을 뿌듯했다는 어떤 성과를 얻어갈 수 있어서 좋았다! 분명 어려운걸 해결하고 나면 뭔가 좋아야만 하는데, 다음에 있을 더 어려운 녀석이 분명 존재할거란 생각에 두려움도 살짝 살짝 다가오는 것 같다..ㅜㅜ

금일 Python Unit test에 대한 주제로, 특강을 들었는데, 어제와 마찬가지로 강사님께서 (특히, 비전공자인 나에게) 좋은 말을 많이 해주셔서 귀에 쏙쏙 박히는 강의였던 것 같다. 더불어, Unit test에 대한 중요성을 한 번 더 깨닫게 되었고, 내 스스로를 업그레이드 하려면 필수적인 요소임을 직감할 수 있었다.

매일매일이 꽉꽉 차있는 느낌이라 뿌듯하기도 하고, 한편으로는 벌써 지치기도 하지만, 나를 발전시키는 시간이라고 생각하며 최선을 다하고 있다! 지치지 않도록 페이스 조절 하면서 열심히 나아가도록 하자! 다른 캠퍼분들도 모두 화이팅~

## 5. 해야할 일
- 1 epoch에서 이뤄지는 모델 학습 과정을 정리해보고 성능을 올리기 위해서 어떤 부분을 먼저 고려하면 좋을지 논의해보자.
- optimizer.zero_grad()를 안하면 어떤 일이 일어날지 그리고 매 batch step마다 항상 필요한지 논의해보자.
- torchvision dataset 클론 코딩 해보기
- DataLoader에서 사용할 수 있는 각 sampler들을 언제 사용하면 좋을지 논의해보자.
- 데이터의 크기가 너무 커서 메모리에 한 번에 올릴 수 없을 때 Dataset에서 어떻게 데이터를 불러오는 게 좋을지 논의해보자.
- Epic vs Epoch?
- Day 11에서 확인헀던 Templete 이해하기!
- OOP에 대한 내용 복습
