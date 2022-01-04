---
layout: post
title: "Transformer"
categories: booststudy
tags: paper
comments: true
---
Attention Is All You Need (Transformer)에 대하여 설명한다.

본 내용은 Attention Is All You Need, 'https://github.com/yukyunglee/Transformer_Survey_Study', Boostcamp Transformer 강의, 동빈나님의 딥러닝 논문 리뷰 Youtube(https://www.youtube.com/watch?v=AA621UofTUA)를 참고하여 작성하였습니다.

## 목차
- [1. Transformer 이전의 모델](#1-transformer-이전의-모델)
- [2. Transformer](#2-transformer)
- [3. 해야할 일](#3-해야할-일)

## 1. Transformer 이전의 모델
### 1-1. 기존 Seq2Seq 모델의 한계점
- Context Vector v에 소스 정보를 압축하기 때문에 다음 그림과 같이 병목 현상이 발생하여 성능 하락의 원인이 된다.
![동빈나 1](https://user-images.githubusercontent.com/53552847/129984486-e405d41f-c52f-4f9f-b1aa-0ccdc2943f0c.png)
- 문제 상황
    - 하나의 문맥 벡터가 소스 문장의 모든 정보를 가지고 있어야하므로 성능이 저하된다.
- 해결 방안
    - 매번 소스 문장에서의 출력 전부를 입력으로 받을 수 있다면?! 가능하다. -> 최신 GPU는 많은 메모리와 빠른 병렬 처리를 지원해줌으로서 어느정도 해결가능하게 되었다.

### 1-2. Seq2Seq with Attention
![동빈나 2](https://user-images.githubusercontent.com/53552847/129984689-f1b316b3-236b-4203-95a0-32c6110ec83e.png)
- Context Vector만 고려하는 것이 아닌 모든 단어를 전부 고려한 하나의 weighted sum 벡터를 구한 후, 이를 v와 함께 입력으로 넣어서 소스 문장(입력)에 대한 모든 정보를 고려하게 만들기 때문에 성능을 높일 수 있다.

### 1-3. Seq2Seq with Attention에서의 Decoder
![동빈나 3](https://user-images.githubusercontent.com/53552847/129984474-e28d8a6f-5ab1-467f-9e55-e0cec191981d.png)
- 위 그림에서 처럼, Attention을 활용할 경우 Attention Weight를 사용해 각 출력이 어떤 입력 정보를 참고했는지 알 수 있으며, 이 가중치를 활용하여 시각화도 어느정도 가능하다.(가중치를 활용해 어떤 정보가 많이 활용되었는지 알 수 있다.)

## 2. Transformer
- 2021 기준 현대 자연어 처리 네트워크에서의 핵심
- RNN, CNN을 전혀 필요로하지 않는다.
- Attention에만 전적으로 의지한다.
- RNN, CNN을 사용하는 대신 Positional Encoding을 활용하는데, 이는 각 단어의 순서에 대한 정보를 입력하기 힘들기 때문이다.
- BERT, GPT와 같은 향상된 네트워크에서 Transformer가 사용된다.
- Encoder, Decoder로 구성되어 있으며, Attention 과정을 여러 레이어에서 반복한다.

### 2-1. 입력 Embedding(Transformer Architecture)
- 입력데이터를 Embedding하는 작업을 의미하며, 전통적인 임베딩 기법을 활용한다.
    - 단어의 개수 만큼의 행의 개수를 가진다.
    - 각 열의 데이터는 Embedding 차원(Hyper Parameter로서 우리가 입력해주는 값이며, 논문에서는 이를 512로 설정하였다.)과 같은 크기의 배열을 사용하게 된다.
- RNN을 사용하지 않으면서 위치정보를 필요로 하므로 위치정보를 포함하고 있는 Embedding을 사용한다.
    - Positional Encoding을 사용한다.
    - 이러한 Positional Encoding은 Input Embedding Matrix와 element-wise 합으로 최종 Input Embedding Matrix가 만들어진다. (마지막, Positional Embedding에서 자세히 다루도록 한다.)
    - 이렇게 함으로서, 각 단어가 어떤 순서를 가지고 있는지에 대한 정보를 Network가 알 수 있도록 만든다.
- 이러한 전반적인 Embedidng 과정이 끝난 후에 Attention을 진행한다.

### 2-2. Encoder
- Attention은 전반적인 입력 문장에 대한 문맥에 대한 정보를 잘 학습하도록 만든다.
- 성능 향상을 위해 Residual Learning을 사용한다.
    - Residual Learning : 입력값을 layer를 거쳐 반복적으로 단순하게 갱신하는 것이 아닌, 특정 layer를 건너뛰어서, 복사된 값을 갱신된 값과 합쳐주는 기법이다.
- Residual Learning을 활용함으로서 네트워크는 기존 정보를 입력받으면서, 추가적으로 Residual된 부분만 학습하도록 만들기 때문에 전반적인 학습 난이도를 낮추고, 이로 인해 초기 모델 수렴 속도를 높여주고, Global Optima를 찾을 수 있는 확률을 높여준다.
![동빈나 4](https://user-images.githubusercontent.com/53552847/129984478-7fe7dd12-fc92-412c-a71c-eee4b365a865.png)
- 위의 그림처럼, Input Embedding Matrix에 Positional Encoding을 element-wise하게 합쳐주고, 이를 Residual Part와 Multi-head Attention part로 보내주어 다시 합치고, Normalization을 진행한다.
- Encoder에서는 위의 과정을 반복하도록 만들어준다.
![동빈나 5](https://user-images.githubusercontent.com/53552847/129984479-9956e3e5-3230-4e4d-b0b6-935c181cf8ea.png)
- 각 Layer는 서로 다른 Parameter를 가진다.
- 각 Layer에서의 입출력의 Dimension은 동일하다.(여러 Layer를 중첩해서 사용하는 것으로부터 유추할 수 있다.)

### 2-3. Encoder to Decoder & Decoder
![동빈나 6](https://user-images.githubusercontent.com/53552847/129984480-a5881399-aa9b-4fd1-9e4b-8f710212dff5.PNG)
- 위 그림처럼, Encoder의 마지막 layer에서 나온 출력값이 Decoder에 input으로 들어가게된다.
- 이러한 과정을 진행하는 이유는, Decoder Part에서 매번 출력할 때마다 소스 문장(Encoder에서의 입력) 중에서 어떤 단어에 가장 많은 초점을 두어야 하는 지를 알 수 있다.
- Decoder Part 역시 여러개의 layer로 구성되어 있다. 더불어, 마지막에 출력되는 값이 우리가 실제로 번역을 실행한 결과값이 된다. 이 때, 각각의 모든 layer들은 Encoder의 마지막 출력값을 input으로 받게된다.
![동빈나 7](https://user-images.githubusercontent.com/53552847/129984481-14c7d232-7471-466c-aad5-b6fe8ac8eeb8.PNG)
- 더불어, 위의 그림처럼, Encoder와 Decoder의 layer의 개수는 동일하게 맞춰주는 경우가 많다고 한다.
- Decoder 역시, Encoder와 마찬가지로 각 단어의 상대적인 위치에 대한 정보를 알려주기 위해 Positional Encoding을 입력에 추가한다.
- 하나의 Decoder layer에는 두 개의 Attention을 사용한다.
- 첫 번째 Attention의 경우 Self-Attention으로, Encoder Part와 마찬가지로 각각의 단어들이 서로가 서로에게 얼마정도의 가중치를 가지는지 구하도록 만들어서 출력되는 문장에 대한 전반적인 표현을 학습한다.
- 두 번째 Attention의 경우, Encoder에 대한 정보를 Attention할 수 있도록 만든다. Decoder에서의 각 출력 단어가 Encoder의 출력 정보를 받아와 이를 사용할 수 있게 만들어준다. 즉, 각 출력 단어가 소스 문장에서의 어떤 단어와 연관성이 있는지를 구해준다. (이를 Encoder-Decoder Attention이라고 한다.)
- Decoder layer 역시 입출력 dimension이 동일하다.
- Encoder는 RNN과 달리 한 번에 모든 인풋을 넣기 때문에 일반적으로 RNN보다 산복잡도가 더 낮다.

### 2-4. Attention
- Encoder, Decoder는 Multi-Head Attention layer를 사용한다.
- Attention을 위한 세 가지 입력 요소는 Q(Query), K(Key), V(Value)이다.
- Q는 무언가를 물어보는 주체로서, 어떤 단어가 다른 단어들과 어느정도의 연관성을 가지는지 구해주는데 이 때, '어떤 단어'의 역할이다.
- K는 물어 보는 대상으로서, 위의 문장에서의 '다른 단어'들의 역할을 한다.
- 예를 들어, I love you라는 단어가 있을 때, MHA에서 Q,K를 활용해 I가 I, love, you 각 단어들과의 어느 정도의 연관성이 있는지, love가 I, love, you 각 단어들과의 어느 정도의 연관성이 있는지, you가 I, love, you 각 단어들과의 어느 정도의 연관성이 있는지를 나타내게 되는데, 이 때, 비교하고자 하는 I가 Q가 되고 비교 대상인 I, love, you가 Key가 된다.
- 이렇게 Q와 K로부터 Score값을 구한 후, 이를 Value와 weigthed sum을 하여 결과값을 추출한다.
- MHA(Multi-Head Attention)이라는 것은, 입력값(Input Embedding Matrix + Positional Encoding)이 들어왔을 때 Q, K, V로 구분되는데 이를 h(head의 개수를 의미)개의 서로 다른 Q, K, V로 구분될 수 있도록 만들어 준다. 이를 통해, h개의 서로 다른 Attention Concept를 학습하도록 만들어 더욱 다양한 특징을 학습하도록 유도한다.
- Q, K를 활용하여 Score값을 만들 때, scaling을 위해 $$\sqrt{d_k}$$ 로 나눠주게 되는데, 이는 Softmax 함수에서 양 끝점에서의 Gradient Vanishing의 문제를 해결하기 위해서 사용하는 것이다. 

### 2-5. Transformer의 동작 원리(단어가 하나만 있다고 가정하고 아래 설명을 진행한 것이다.)
- Attention을 위해 Q, K, V가 필요하다.
- 각 단어의 임베딩을 이용해 이를 생성할 수 있다.
    - Embedding 차원을 $$d_{model}$$ 이라고 하자. 
    - Q, K, V는 $$d_{model}$$ / h 만큼의 차원으로 구성된다.
    - 만약, Embedding이 4차원이고 h가 2개일 경우, Q, K, V의 차원은 2가 된다.
![동빈나 8](https://user-images.githubusercontent.com/53552847/129988125-009edbcb-8198-497e-8a9e-e7c52808283f.png)
- Attention value의 값은 Q, K, V와 같은 차원을 가진다.(?)
- Mask Matrix를 활용하여 특정 단어의 가중치를 0으로 만듬으로서 특정 단어를 무시할 수 있다.
- $$QK^T$$ 와 같은 크기의 Mask Matrix를 만들게 되는데, 이 때 마스크 값으로 부터 -inf의 값을 가지게 만들어, softmax를 취할 때 출력을 0에 가깝게 만들게 된다.
- Multi-Head Attention을 수행한 뒤에도 차원이 동일하게 유지된다.
- Transformer에는 다음의 세 가지 종류의 Attention layer가 사용된다.
    - Encoder Self-Attention
        - 각각의 단어가 서로에게 어느정도의 연관성을 가지는지 Encoder를 통해서 구하게 된다.
        - 전체 문장에 대한 Representation을 Learning하도록 한다.
    - Masked Decoder Self-Attention
        - Encoder Self-Attention과 거의 동일하지만, Decoder의 경우 출력 단어를 뽑아낼 때 뒤의 단어를 input으로 사용하는 것은 cheat이므로, 출력 단어가 앞에 등장한 단어만 참고할 수 있도록, 뒤의 단어를 Mask한 형태로 Self-Attention을 진행한다.
    - Encoder-Decoder Attention
        - Decoder의 Query값이 Encoder의 Key, Value를 참고하여 Attention을 진행한다. 
- Self-Attention
    - Encoder, Decoder 모두에서 사용된다.
    - 매번 입력 문장에서 각 단어가 다른 어떤 단어와 연관성이 높은지 계산한다.
    - Attention Score를 통해 시각화를 어느 정도 할 수 있다.

### 2-6. Positional Encoding
- 주기 함수를 활용한 공식을 사용한다. (사실상, 주기 함수가 아닌 학습이 가능한 형태라면 어떤 것을 사용하여도 성능상의 큰 차이는 없다.)
- 각 단어의 상대적인 위치 정보를 Network에 입력한다.
![동빈나 9](https://user-images.githubusercontent.com/53552847/129988126-21a7758e-bf18-4c7a-8371-b56ec1bc2023.png)
- 위 그림에서, pos는 각 단어의 번호를 의미하며, i는 Embedding 값의 위치를 의미한다.
- 위 그림과 같이, 일반 임베딩에 Positional Encoding값을 더해서 Attention에 활용하게 된다. 더불어, Positional Encoding값은 일반 임베딩에서의 공식을 활용하여 만들게 된다.

## 3. 해야할 일
- Residual Learning을 사용하면 학습난이도가 낮고, 초기 모델 수렴속도가 높고, Global Optima를 찾을 확률이 높아진다고 하는데 왜그럴까?
- 각 Layer에서의 입출력의 Dimension은 동일하다.(여러 Layer를 중첩해서 사용하는 것으로부터 유추할 수 있다.) -> 어떻게 유추하는 걸까?
- Decoder의 초기 input값은 무엇인가?
- Q, K, V가 각 Head마다 다르게 만들어, Attention Concept을 학습하도록 한다는 것은 Q, K, V는 만들어 줄 때마다 계속해서 바뀐다는 것인가? 
- Attention value의 값은 Q, K, V와 같은 차원을 가진다.(?) -> Q, K는 같은 차원을 가지고, V는 다른 차원을 가져도 된다고 하지 않았는가?
- Decoder에서의 Query가 Encoder에서의 K, V를 참고하여 연산하게 되는데, 이 때, Query와 Key의 차원이 달라도 된다. 그렇다면 내적은 어떻게?
