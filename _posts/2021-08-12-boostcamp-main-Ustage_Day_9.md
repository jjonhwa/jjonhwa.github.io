---
layout: post
title: "Ustage Day 9"
categories: boostcamp
tags: main
comments: true
---
RNN과 Transformer에 대한 설명을 진행한다.

**부스트 캠프 9일차 학습 요약**
- **행사** : Git 특강, 오피스 아워
- **학습** : RNN, Transformer
- **피어 세션**

## 목차
- [1. Sequential RNN](#1-sequential-rnn)
- [2. Transformer](#2-transformer)
- [3. Multi-Headed Attention](#3-multi-headed-attention)
- [4. 피어 세션](#4-피어-세션)
- [5. 9일차 후기](#5-9일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Sequential RNN
- Sequential Model의 정의와 종류에 대하여 배운다.
- RNN에 대한 정의와 종류에 대하여 배운다.

### 1-1. Sequential Model
- Naive Sequence Model
- Sequential Data를 처리하는데 가장 큰 어려움은?
    - Sequential Data는 정의상 데이터의 길이가 얼마나 될지 알 수 없다.
    - Input의 차원을 기본적으로 알 수 없다. 그러므로 Fully Connected Layer과 CNN을 사용할 수 없다.
    - 시간이 지남에 따라 고려해야할 과거의 정보량이 점점 늘어난다.
- Markov Model (First-Order Autoregressive M0del)
    - 가정을 현재는 과거에만 의존적이라고 정의한다.
    - First-Order를 사용하게되면 Joint Distribution을 표현하기가 많이 쉬워진다.
    - Decoder에서 사용한다.
- Latent Autoregressive Model
![DL Basic 24](https://user-images.githubusercontent.com/53552847/129213936-e109c3a9-d1d6-4006-b4a8-842438658c39.PNG)
    - 위의 그림과 같이 중간에 Hidden State를 삽입함으로서 과거 정보를 삽입할 수 있도록 만든다.

### 1-2. Recurrent Neural NEtwork
- $$X_t$$에만 의존하는 것이 아니라 t-1에 얻어진 어떤 Cell State에 Dependent하게 된다.
![DL Basic 25](https://user-images.githubusercontent.com/53552847/129213946-f889d2f7-929a-4643-ba57-9110ff0998bd.PNG)
- RNN은 Time Step을 고정한 후에 시간순으로 순환된 것을 풀게되면 결국 각 Network가 Parameter들을 Share하는 굉장히 Input이 큰 네트워크 하나가 된다.
- 학습의 어려운 점은 없는가?
    - Short-Term Dependencies의 문제가 있다.
    - 먼 과거에 있는 정보가 미래까지 살아남기 쉽지 않다.
    - 가까운 과거의 정보는 잘 고려가 되지만, 먼 과거가 잘 고려되지 않는다.
![DL Basic 26](https://user-images.githubusercontent.com/53552847/129213948-f7c95d6c-96d4-4658-8956-a39fe921cfc8.PNG)
    - 위의 그림처럼 $$h_0$$는 가중치들의 곱으로 계속 중첩되게 되는데, 이 때 Activation Function이 sigmoid라고 생각했을 경우에는 점점 0으로 수렴되는 현상이, ReLU라고 생각했을 때는 값이 폭발적으로 커짐을 알 수 있다.
    - 이러한 문제 때문에 RNN을 사용하는 것도 문제가 있으며 LSTM, GRU가 등장하였다.

### 1-3. LSTM (Long Short Term Memory)
![DL Basic 27](https://user-images.githubusercontent.com/53552847/129213950-312b4eff-8bde-485c-9fab-505aa9c0907d.PNG)
- Cell State, Hidden State, Input이 input으로 들어오게 되고, Output, Cell State, Hidden State가 Output이 된다.

![DL Basic 28](https://user-images.githubusercontent.com/53552847/129213952-df2b619b-b575-4ee8-9fa4-7c3fdc73e9f9.PNG)
- 위의 그림처럼 LSTM은 3개의 Gate로 이루어져 있다.
#### Forget Gate
![DL Basic 34](https://user-images.githubusercontent.com/53552847/129216409-f562b444-cd41-4d0b-a748-fe4cc540a08f.png)
    - 어떤 정보를 버릴지 결정한다.
    - 현재의 입력 $$X_t$$와 이전의 Hidden State Output $$H_{t-1}$$ 이 들어가서 $$f_t$$라는 0~1 사이의 Scalar값을 가지게 된다.
![DL Basic 29](https://user-images.githubusercontent.com/53552847/129213955-66de7bed-53d3-4d2e-a3db-fb5e5b2c00b7.PNG)
    - 이전의 Cell State에서 나온 정보 중에 어떤 것을 버리고 어떤 것을 살리릴지 결정한다.

#### Input Gate
- Cell State에다가 정보를 올리는데, 이 정보 중에 어떤 정보를 올릴 것인지 정한다.
![DL Basic 35](https://user-images.githubusercontent.com/53552847/129216412-672b7488-c89e-40eb-88bd-f4ee0837f9e4.png)
![DL Basic 30](https://user-images.githubusercontent.com/53552847/129213957-4a00304f-a49d-481b-b28f-c783452f6e38.PNG)
- $$i_t$$는 어떤 정보를 추가하고 안할지 결정한다.
- $$\tilde C_t$$는 올릴 정보들을 의미한다.

#### Update Gate
![DL Basic 36](https://user-images.githubusercontent.com/53552847/129216413-88b9b87b-6db5-41bb-8257-cf420dc05533.png)
- $$\tilde C_t$$를 $$i_t$$만큼 곱해서 어떤 정보를 올릴지 결정한다.
- 이전 Cell State에서 들어온 값을 $$f_t$$와 곱해서 어떤 정보를 올릴지 결정한다.
- 위의 두 값을 합쳐서 새로운 Cell State를 만들어준다.
![DL Basic 31](https://user-images.githubusercontent.com/53552847/129213961-edc3714a-297d-46aa-9068-bd5231ef6b52.PNG)

#### Output Gate
![DL Basic 37](https://user-images.githubusercontent.com/53552847/129216415-870590ee-dfe3-4333-996d-23b7e13f0b6a.png)
- 출력되는 Cell State를 한 번 더 조작한다.
- 어떤 값을 밖으로 내보낼 지 Output Gate로부터 값을 구하고, 그 만큼 Cell State로 부터 나오는 값에 곱해서 Output을 출력한다.
![DL Basic 32](https://user-images.githubusercontent.com/53552847/129213966-dd4997ca-edf2-422a-90c6-18c033d67f11.PNG)

### 1-4. GRU
![DL Basic 33](https://user-images.githubusercontent.com/53552847/129213968-1b0993ef-848b-4740-9d38-3dd9e96f4cee.PNG)
- Gate가 2개 있다. (Reset Gate, Update Gate)
- Cell State가 없고, Hidden State가 Output이자 다음 Cell의 Input Hidden State이다.
- LSTM에 비해 Gate가 줄어서, Parameter의 수가 줄어들었다.
- 적은 Parameter로 동일한 Output을 내면, Generalization Performance가 증가한다는 점에서 LSTM보다 더 좋은 경우가 많다.


## 4. 피어 세션
### 4-1. [새로운 캠퍼님에게 그라운드룰 소개]

- 모더레이터 순서에 대해서 소개(강진선 -> 김범수 -> 박승찬 -> 심우창 - 우원진 -> 최성욱 -> 배민환)
- 코드리뷰를 위한 github 초대
 
### 4-2. [이전 질문 리뷰]
- 이전 시간에 대한 질문은 당일날 해결해서 생략

### 4-3. [금일 질문 목록]:
- 선택과제 3번(승찬님이 하시다가 막히거나 잘 이해되지 않았던 부분)
- Gaussian mixture를 왜 사용하는지 그리고 어떻게 사용해야 하는지에 대해서 궁금하다.
- 주어진 MDN class를 참고하여 과제를 수행했다. <- 이상한 값이 섞여서 나왔다.
    - 이유 : gaussian을 제대로 이용하지 못해서 그런 것 같다.
- 선택과제 3번 추가 질문
    - foward부분에서 shape를 어떻게 맞춰줬는지?
        - hidden layer로 들어가서 n_gaussians으로 나오기 때문에 이 부분을 통해서 shape를 맞춰줬다(코드상에 맞춰져 있었다).
    - 코드상에 x.mm이 무엇을 의미하는지? 
        - 행렬곱인것 같다.
- 선택과제 3번 참고 자료
    - 진선님이 참고하신 블로그(https://mikedusenberry.com/mixture-density-networks)
- '필수과제(MHA)에서 Q,K,V의 개수에서 K와 V는 같아야 되지만 Q는 달라도 된다'라고 하셨는데 이 부분이 잘 이해가 되지 않는다.
- 선택 과제에 대해서 해설 듣고 거기서도 이해가 안 되면 주말에 찾아보고 그것으로도 해결이 안되면 멘토님에게 물어보기 
- LSTM에서 update cell이랑 output gate가 이해가 잘 안되었습니다. 
    - forget gate : 이전의 hidden state와 가중치를 시그모이드 해준 결과이다.
    - input gate : 전 hidden state와 가중치를 시그모이드 해주고. hidden state에서 
    - update cell : 이전 cell state에서 다음으로 전달할 cell state구하는 방법으로 과거에 잊어버릴것(ft(이전 hidden state 시그모이드를 거친 값) * Ct-1(이전 cell state))은 잊어버리고 새로 기억할것(it(이전 hidden state 시그모이드를 거친 값) * Ct(hidden state가 tanh를 거친 값))은 기억하자
    - output gate : ot(이전 hidden state에서 시그모이드 거친 값) * (다음 cell state에서 tanh를 거친값)이다.
 
### 4-4. [선택과제 정답 살펴보기]

- 선택과제 3 - softmax함수 인자로 dim을 주는게 어떤 의미인가? 
softmax연산을 해당 dim을 기준으로 한다.
- 선택과제 1 - residual부분 -> 통합적인 구조를 만들고 싶을때 사용
- 선택과제 1 - attention list부분
- 선택과제 1 - encoder부분에서 MHA 후 norm하기 전 원래값 더해주는 부분

## 5. 9일차 후기


## 6. 해야할 일
- Further Question : LSTM에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있다. 무었일까?
- Further Question : Pytorch LSTM 클래스어서 3 Dim 데이터(Batch_size, Sequence Length, Num Feature), Batch_First 관련 Argument는 중요한 역할을 한다. batch_first=True인 경우에는 어떻게 작동하게 되는 걸까??

