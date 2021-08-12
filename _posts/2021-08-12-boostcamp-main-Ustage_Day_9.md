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
- [3. 피어 세션](#3-피어-세션)
- [4. 9일차 후기](#4-9일차-후기)
- [5. 해야할 일](#5-해야할-일)

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

## 2. Transformer
- Sequential Model의 한계점.
- Encoder와 Multi Head Attention에 집중한다.
![DL Basic 38](https://user-images.githubusercontent.com/53552847/129281925-e7d0d194-879f-4f9e-8364-1d26111fccb9.png)

### 2-1. Sequential Model
- Sequential Modeling은 다루기에 힘들다.
    - Trimmed, Ommited, Permuted와 같은 이유 때문.
    - 이런 문제를 해결하고자 등장한 것이 Trnasformer이다.

### 2-2. Transformer
- 재귀적인 구조가 없고, Attention이라는 구조를 활용한다.
- Sequential한 데이터를 처리하고 Encoding하는 방법이다. -> NMT 문제 뿐만 아니라 다양한 문제 해결에 사용된다.
- 동일한 구조를 가지지만  Network Parameter가 다르게 학습되는 Encoder와 Decoder의 구조를 가진다.
- 'N개의 단어가 어떻게 Encoder에서 처리 될까?', 'Encoder와 Decoder 사이에 어떤 정보를 주고 받는가?', 'Decoder가 어떻게 Generation을 할 수 있을까?'. 이 세가지가 Transformer을 이해하는데 있어 핵심 질문이다.

### 2-3. N개의 단어를 어떻게 Encoder에서 처리할 수 있을까?
- 하나의 Encoder는 Self-Attention과 Feed Forward Neural Network로 이루어져 있다.
- Transformer가 잘 작동할 수 있게 만든 것은 Self-Attention의 구조 덕분이다.
- Feed Forward Neural Network는 MLP와 동일하다.
- Self-Attention을 지나면 $$X_1$$이 $$Z_1$$이 되는데, 이 때 나머지 X들의 정보도 함께 전달한다. 즉, 각 input vector들이 self-attention을 통과할 때, 다른 Input Vector들도 함께 사용하게 되는 것이다.
- Self-Attention은 Dependency가 모두에게 있다.

### 2-4. Encoder의 작동 원리
- 총 3개의 Vector를 만들어 낸다. Query, Key, Value Vector.
- Query, Key, Value는 각 단어(입력)이 주어졌을 때, 각각 생성이 되고,  이 세 개의 벡터를 통해 입력 단어에 대한 Embedding Vector를 새로운 Vector로 바꿔준다.
- 내가 인코딩하고자 하는 벡터의 Query Vector와 나머지 모든 N개의 단어에 대한 Key Vector를 내적한 것이 $$i_{th}$$ 단어(입력)에 대한 Score Vector이다. 이를 통해 $$i_{th}$$ 단어가 나머지 단어들과 얼마나 유사도가 있는지 나타낸다.
- Attention : 특정 태스크를 수행할 때, 특정 Time Step에 어떤 입력들을 더 주의깊게 봐야하는 지를 나타낸다. 즉, 입력(단어)을 Encoding하고 싶은데, 어떤 나머지 단어들과 더 많은 Interaction이 일어나게 만들어야하는지 Query Vector와 나머지 Vector들의 Key Vector들의 내적을 함으로서 알 수 있게된다.
- Score Value가 너무 커지는 것을 방지하기 위해 Normalize를 한다. 이 때, Normalize의 경우, Key Vector의 Dimension에 Dependent하며 Score Vector를 `sqrt(dimension of K)`로 나눠줌으로서 Normalize하게 된다. (Key Vector의 Dimension을 설정하는 것은 Hyperparameter이다.)
- 각 Score Value 대한 Weight를 구하기 위해 Softmax를 취한다.
- 이렇게 구해진 Attention Weight가 Interaction 값이 된다.
- Query와 Key Vector는 내적을 해야하므로 차원이 같아야한다.
- 최종으로 Value Vector와 Attention Weight를 Weighted Sum을 한다. (Weighted Sum이란 가중치끼리의 곱한 후 합한다는 의미이다.)
- Value Vector와 Encoding Vector의 차원은 같아야한다. (MHA에서는 둘의 차원은 달라지는데, 이는 Encoding Vector를 구할 때 쪼개서 구하기 때문이다.)
- Encoder가 잘되는 이유는?
    - 하나의 고정된 Input이 있다고 할 때, 내가 Encoding하려는 단어와, 그 옆에 있는 단어들에 따라서 Encoding된 값이 달라진다.
    - 즉, 좀 더 Flexible한 Model이 된다. -> Output이 고정되지 않고, 훨씬 더 많은 것을 표현할 수 있게된다.
- N개의 단어가 주어지면 NxN Attention Map을 만들어야 하므로 Length가 길어짐에 따라 한계가 발생한다. 하지만, 이로부터 얻어지는 이득이 훨씬 많다.
![DL Basic 39](https://user-images.githubusercontent.com/53552847/129281931-9aea5eb2-8a23-41b1-adc0-78e367f9a666.png)

#### Multi-Headed Attention
- N개의 Attentino을 반복하게 된다.
- Input Vector와 Encoder Vector의 차원이 같아야한다.
- 여러개의 Encoding Vector가 있을 때, 이를 다시 한 번 가중치 행렬과 곱해줘서 Input과 Output이 같게 만들어준다.
- 실제로는 여러번 Encoding을 하는 것이 아니라 하나의 Input을 여러개로 쪼개서 Encoding을 진행한다.
- Bias를 의미하는 Positional Encoding을 추가하는데 이는 다음의 의문으로부터 해결할 수 있다.
    - Self-Attention은 Order Independent하다. (왜냐면, 각 단어가 어느 위치에 있든 다른 모든 단어를 고려하므로)
    - Sequential Data에 있어서 Order는 중요한 요소이다.
    - Order에 대한 값이 추가되어야 하지 않을까?

### 2-5. Encoder to Decoder
- Encoder에서 출력된 Key, Value Vector를 Decoder로 보낸다.
- Decoder로부터 만들어지는 Query Vector와 Encoder로부터 만들어지는 Key Vector, Value Vector를 활용하여 최종값을 만들게 된다.
- 최종 출력은 Auto Regressive(하나씩 출력하고, 과거의 단어만 뒤의 단어에 영향을 준다.)하게 만들어진다.
- 학습할 때에는, 입출력을 알고 있으므로, Decoding에서 Masking을 활용하여 앞에 오는 단어에는 Dependent하고 미래의 정보는 활용하지 않게 학습하도록 한다.
- Decoder에 들어가는 단어들 이전까지의 Generation 단어들만 가지고 Query를 만들고, Input last sequence에서 나오는 K, V를 활용하여 Input Encoded Vector를 활용하게 된다.
- 단어들의 분포를 만들어서 그 중의 단어를 매번 샘플링하는 식으로 돌아간다.

## 3. 피어 세션
### 3-1. [새로운 캠퍼님에게 그라운드룰 소개]
- 모더레이터 순서에 대해서 소개(강진선 -> 김범수 -> 박승찬 -> 심우창 - 우원진 -> 최성욱 -> 배민환)
- 코드리뷰를 위한 github 초대
 
### 3-2. [이전 질문 리뷰]
- 이전 시간에 대한 질문은 당일날 해결해서 생략

### 3-3. [금일 질문 목록]:
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
 
### 3-4. [선택과제 정답 살펴보기]

- 선택과제 3 - softmax함수 인자로 dim을 주는게 어떤 의미인가? 
softmax연산을 해당 dim을 기준으로 한다.
- 선택과제 1 - residual부분 -> 통합적인 구조를 만들고 싶을때 사용
- 선택과제 1 - attention list부분
- 선택과제 1 - encoder부분에서 MHA 후 norm하기 전 원래값 더해주는 부분

## 4. 9일차 후기
Git특강으로부터 협업에 대한 기술을 한 층 더 향상 시킬 수 있었다. 기존에 조금은 빡빡한 공부를 하다가 들어서인지 한 층 수월하게 수강할 수 있었고, 다소 재미있는 요소들이 있었기에 공부하는 데 활력을 더해줄 수 있었던 것 같다.

더불어, 실제로 많이 쓰이는 트렌드에 대한 VIT, AAE, MDN을 맛보고나서 충격이었던 찰나에 Transformer를 배움으로서 조금은 흥미를 더해가며 공부할 수 있었던 것 같다. 이들이 어디에 쓰이는지 알고 배우는 것과 모르고 배우는 것의 차이를 느낄 수 있었다.

당연히, 공부의 난이도는 기하급수적으로 올라가고 있지만 점점 익숙해지고 있는 것 같다. 더불어 주말도 반납해야 되지 않나라는 생각이 머리끝까지 올라왔다 ㅜㅜ.

최선을 다해서 마무리하기로 다짐한 만큼 더더욱 열심히 해야겠다!! 다른 모든 캠퍼분들도 파이팅하시기를!

## 6. 해야할 일
- Further Question : LSTM에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있다. 무었일까?
- Further Question : Pytorch LSTM 클래스어서 3 Dim 데이터(Batch_size, Sequence Length, Num Feature), Batch_First 관련 Argument는 중요한 역할을 한다. batch_first=True인 경우에는 어떻게 작동하게 되는 걸까??
- Input Vector와 Encoder Vector의 차원이 같아야한다. 그렇다면 Value Vector는 Input Vector의 차원과 같게 만들어줘야 하는가?
- Decoder에 들어가는 단어들 이전까지의 Generation 단어들만 가지고 Query를 만들고, Input last sequence에서 나오는 K, V를 활용하여 Input Encoded Vector를 활용하게 된다. -> 무슨 말인가?
- 단어들의 분포를 만들어서 그 중의 단어를 매번 샘플링하는 식으로 돌아간다. -> 무슨 말인가?
