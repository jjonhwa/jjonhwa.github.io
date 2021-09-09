---
layout: post
title: "NLP Ustage Day 2 (Day 26)"
categories: boostcamp
tags: main
comments: true
---
자연어 처리의 가장 간단한 모델에 대하여 학습한다.

**부스트 캠프 26일차 학습 요약**
- **학습** : RNN, Language Model, LSTM & GRU
- **피어 세션**

## 목차
- [1. Recurrent Neural Network](#1-recurrent-neural-network)
- [2. Experiment Toolkig](#2-language-model)
- [3. LSTM & GRU](#3-lstm---gru)
- [4. 피어 세션](#4-피어-세션)
- [5. 26일차 후기](#5-26일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Recurrent Neural Network
### 1-1. Basic Structure
![14](https://user-images.githubusercontent.com/53552847/132627292-30b2e0e9-0b3a-4127-b819-d919ebf30d70.PNG)
![15](https://user-images.githubusercontent.com/53552847/132627294-964cfbbd-5437-474c-8bc9-3a8f2f26a7bd.PNG)
![16](https://user-images.githubusercontent.com/53552847/132627268-73196a2f-bfc8-4c70-96c4-1762acb18dc7.PNG)
- RNN의 가장 중요한 특징으로서 Parameter W는 모든 time-steps에서 동일한 값을 공유한다.

### 1-2. Problem Setting
![17](https://user-images.githubusercontent.com/53552847/132627270-87fc0601-c62a-4d68-a0b6-ba0f5420d83f.PNG)
- One-to-one
    - Standard Neural Network 
    - 입력과 출력 그 어디에도 Sequence data or Time step으로 이루어지지 않은 데이터로 구성되어있는 일반적인 형태이다.
- One-to-many
    - Image Captioning
    - 하나의 이미지가 입력으로 들어가고, 이 이미지에 대한 설명글을 생성한다.
    - 맨 처음음을 제외한 입력값은 맨처음 입력과 동일한 크기이면서 전부 0으로 채워져있는 값을 input으로 넣어준다.
- many-to-one
    - Sentiment Classification
- many-to-many
    - 위의 그림에서 좌측 many-to-many의 그래프는 Machine Translation과 같이 input의 정보를 모두 받아들인 뒤 이를 바탕으로 번역을 하는 태스크이다.
    - 우측 many-to-many의 경우, 어떤 정보의 출력의 delay가 없는 형태로, video classification on frame level 혹은 POS tagging등의 task가 이 형태에 속한다.

## 2. Language Model
- RNN 구조를 활용하여 자연어처리를 할 수 있는 가장 간단한 task인 Character-level Lanuguage Model에 대하여 살표본다.
- Language Model task는 우리에게 주어진 문자열이나 단어들의 순서를 바탕으로 다음 단어가 무엇인지 예측하는 task이다.
- character level의 language model의 경우, 각 단어 다음에 나올 character가 무엇일지 예측해야한다.
- 즉, many-to-many task에 속한다.

### 2-1. Language Model 예측 과정
- 주어진 학습 데이터로부터 중복없이 사전을 구축한다.
- 각각의 Character는 총 사전의 개수만큼의 dimension을 가지는 one-hot vector로 나타낼 수 있다.
- 다음과 같이 각 철자의 one-hot vector가 각 time-step에서의 입력이 된다.
![18](https://user-images.githubusercontent.com/53552847/132627274-d3f5a786-8afb-4996-923d-d2b2736436e4.PNG)

- 위의 그림에서 처음 입력되는 $$h_0$$는 디폴트 값으로 모두 0인 벡터를 입력으로 넣어준다.
![19](https://user-images.githubusercontent.com/53552847/132627275-c38cf88e-9b76-48e3-b5d6-2bd01e40c864.PNG)

- 위의 그림과 같이 최종 output이 Logit이라고 표현된 이유는 다음과 같다.
    - 사전에 정의된 4개의 Character 중 한 character로서 다음의 나올 character를 예측하는 task로서 output layer의 node의 수는 사전의 크기와 동일한 4개의 dimension으로 구성된 output vector가 나오는 것을 알 수 있다.
    - 이렇게 얻어진, output vector를 Multiclass Classification을 수행하기 위해 softmax layer를 통과시키게 되면, output vector가 logit값으로서 제일 큰 값을 가질 때의 확률값에 대한 target만 출력되기 때문이다.
- 이렇게 출력된 target이 ground truth의 값과 동일하게 만들어지도록 학습이 진행된다.

![20](https://user-images.githubusercontent.com/53552847/132627276-39e265ff-3d58-414c-8205-e82a2c6b1704.PNG)
- inference를 진행할 경우, 첫번째 character만을 입력으로 주게 되는데 이렇게 해서 얻은 output을 다음 time step의 입력값으로 넣어 예측을 반복적으로 진행하게 된다.
- 이를 바탕으로 멀리 떨어진 미래에 대한 값도 RNN을 통해 예측할 수 있게 된다.

### 2-2. Shakespeare's plays
- character level의 language modeling을 단순히 하나의 단어가 아닌 보다 긴 문단에 대해서도 학습할 수 있다.
- 문단에서 나타나는 character들의 sequence는 RNN을 학습하는데 사용되는 학습데이터로서 활용될 수 있다.
- 문단의 경우, 여러 단어로 이루어져 있고 여러 문장으로 이루어져 있기 때문에 character level의 language modeling을 하기 위해서는 공백 역시 하나의 특수문자로서 vocabulary 상의 하나의 character로 보게된다.
- 가령 쉼표, 줄바꿈 역시 사전에 기록해두고 사용해야 한다.
- 이렇게 하면, 하나의 글을 여전히 one dimensional character sequence로 볼 수 있고, 이를 통해 language model을 학습할 수 있게 된다.
![21](https://user-images.githubusercontent.com/53552847/132627277-e17eeaf3-40c4-4992-90ff-ca39b67916e8.PNG)

### 2-3. BPTT (Backpropagation through time)
- Forward through entire sequence to compute loss, then backward through entire sequence to compute gradient
- 현실적으로 Sequence의 길이가 너무 길어지면 한꺼번에 처리할 수 있는 데이터의 양이 한정된 GPU resource에 담기지 못할 수 있기 때문에 truncation 즉, 군데 군데 잘라서 제한된 길이의 sequence만으로 학습을 진행하는 방식을 채택한다.

### 2-4. Vanishing / Exploding Gradient Problem in RNN
- Original RNN의 구조에서 $$W_{hh}$$ 의 구조가 반복적으로 반영된다는 사실 때문에 흡사 등비수열의 방식의 backpropagation이 진행되는데, 이 때, $$y_t$$에서 발생한 gradient가 처음으로 돌아 갈수록 기하급수적으로 크거나 작아질 수 있다.
- Toy Example로서 다음을 이해해보자.
![22](https://user-images.githubusercontent.com/53552847/132627279-6a27330c-abc6-4af3-82ab-2ea27a0ee808.PNG)
- 위의 그림에서 볼 수 있듯이, $$W_{hh}$$ 가 3인데 이 값이 Backpropagation을 통해 진행되면 진행될 수록 중첩하여 쌓이므로 더 옛날 값으로 갈 수록 이 값이 기하급수적으로 커지는 것을 알 수 있다.
- 즉, 일반적인 Vanilla RNN을 사용하게 되면, 뒤쪽의 time step까지 유의미한 Gradient Signal을 보낼 수 없게 된다.

## 3. LSTM & GRU
### 3-1. LSTM (Long Short-Term Memory)
- Core Idea : pass cell state information straightly without any transformation
- Solving long-term dependency problem
![23](https://user-images.githubusercontent.com/53552847/132627280-6f9d6d51-9fef-43b4-aca1-246f2e4a9390.PNG)
- 기존의 Vanilla RNN의 식이 $$h_t = f_t(x_t, h_{t-1})$$ 이었다면, LSTM은 $$(c_t, h_t) = LSTM(x_t, c_{t-1}, h_{t-1})$$ 이 된다.
- 이 때, 위 식에서의 $$c_{t-1}$$ 을 cell state vector, $$h_t$$ 을 hidden state vector라고 한다.
- Cell state vector  vs  Hidden state vector
    - cell state vector가 더욱 완성된, 더 많은 정보를 담고 있는 vector이다.
    - hidden state vector는 cell state vector를 한 번 더 가공해서 해당 time step에서 노출할 필요가 있는 정보만을 남긴 filtering된 정보를 담는 vector라고 이해할 수 있다.
    - hidden state vector는 현재의 해당 time step에서 예측값을 계산하는 output layer등의 다음 layer의 입력벡터로 사용한다는 특징이 있다.

### 3-2. LSTM의 연산과정
![24](https://user-images.githubusercontent.com/53552847/132627282-d6824340-c1ba-42dc-86dd-fa91d2904cf0.PNG)
- $$x_t, h_{t-1}$$ 을 입력으로 받는다.
- 이런 input vector들을 W를 바탕으로 선형변환한다.
- 이렇게 선형변환된 input vector는 4개의 벡터들로 분할하게 되는데, 각각의 벡터 원소별로 sigmoid를 거치고(마지막 벡터만 tanh를 거친다.) 이를 바탕으로 output vector를 생성하게 된다.
- 이렇게 생성된 output vector를 각각 input gate, forget gate, output gate, gate gate라고 부른다.
- 이렇게 생성된 input, forget, output, gate gate들이 cell state 및 hidden state를 최종적으로 계산하기까지 필요로 하는 중간 결과물로서의 역할을 하게 된다.
- 각 gate의 세부역할을 다음과 같다.
    - input gate : whether to write to cell
    - forget gate : whether to erase cell
    - output gate : how much to reveal cell
    - gate gate : how much to write to cell
- sigmoid의 특성상 i, f, o의 output은 모두 0~1 사이의 값을 가진다.
- sigmoid를 통해 나온 vector들은 어떤 다른 vector와의 element wise multiplication을 통해 원래 값에서의 일부만 가지도록 하는 역할을 한다. 즉, 기존 값에서 sigmoid percentage 만큼만 값을 보존하게 된다고 이해할 수 있다.
- tanh를 통해서 나온 값은 특성상 -1 ~ 1 사이의 값을 가지게 되는데, 이는 original RNN에서 선형결합 후 tanh를 통해서 최종 hidden state vector를 -1 ~ 1의 값으로서 어떤 유의미한 정보를 담는 역할을 했듯이 마찬가지로 tanh를 지나는 값이 현재 time step에서 LSTM에서 계산되는 유의미한 정보라고 이해할 수 있다.

### 3-3. LSTM Gate의 역할
- A gate exists for controlling how much information could flow from cell state
- 즉, 기본적으로 gate를 통과한 vector들은 전 time step에서 넘어온 cell state vector인 $$c_{t-1}$$ 을 적절히 변환하는데 사용한다고 이해할 수 있다.
- Forget gate
![25](https://user-images.githubusercontent.com/53552847/132627284-d3c879ee-98d8-4670-bb59-82ab46dccfc0.PNG)
    - forget gate vector의 차원은 $$c_{t-1}$$ 의 차원과 동일하다.
    - forget gate vector와 $$c_{t-1}$$을 element wise multiplication을 진행한다.
    - 이런 연산과정을 통해, 이전 time step에서 넘어온 cell state vector에서 sigmoid 값 만큼만을 보존해준다.
- Gate gate & Input gate
![26](https://user-images.githubusercontent.com/53552847/132627286-d6f90c08-c908-4e2c-ac55-c6340b2564fe.PNG)
    - $$\tilde C_t$$ 가 gate gate이고, $$i_t$$가 input gate이다.
    - $$\tilde C_t$$ 가 현재 input에서 유의미한 정보를 담고 있는 역할을 한다.
    - 이를 바로 forget과 연산된 $$c_{t-1}$$ 에 더해주지 않고 input gate와 multiplication을 진행한 후에 더해준다.
    - 이렇게 하는 이유는, 가령 한 번의 선형변환 즉 tanh 만으로 이전의 cell state에 더해주고자 하는 정보를 만들기가 어려운 경우, 기존에 더해주고자 하는 값보다 더 크게 $$\tilde C_t$$ 를 만들어주고 이 값을 input gate의 비율만큼으로 정보를 덜어내 정보를 전달하기 위해서이다.
- Output gate
![27](https://user-images.githubusercontent.com/53552847/132627289-01c77d2c-42bc-4ac6-b052-90f006dde94e.PNG)
    - Generate hidden state by passing cell state to tanh and output gate
    - Pass this hidden state to next time step, and output or next layer if needed
- Original RNN에서의 hidden state vector와 같이 -1 ~ 1 사이의 값의 범위를 가지는 벡터로 $$c_t$$ 를 변형해주고 이를 output gate로부터 나온 sigmoid vector와 곱해줌으로서 최종 hidden state vector를 만들어준다.
- 즉, cell state가 가진 정보에서 특정 dimension 별로 적절한 비율만큼으로 이 값들을 작게 만들어 $$h_t$$를 출력하게 된다.
- $$c_t$$는 기억해야할 모든 정보를 담고 있는 벡터라고 이해할 수 있고, $$h_t$$는 현재 time step에서 예측값을 내는 output layer의 입력으로 사용되는 벡터이다.
- 즉, 해당 time step의 예측값의 직접적으로 필요한 정보만을 담고 있으며, $$c_t$$가 가지는 많은 정보들 중에서 지금 당장 필요한 정보만을 filtering한 형태로 이해할 수 있다.
- $$h_t$$는 다음 RNN cell로 넘어감과 동시에 현재 time step에서의 예측을 수행하는 output layer의 입력으로 들어간다.

### 3-4. GRU (Gated Recurrent Unit) 
![28](https://user-images.githubusercontent.com/53552847/132627291-85fd2edb-7e31-45d5-8510-a3ac5b7ef5db.PNG)
- LSTM과 비교하여, LSTM의 모델 구조를 보다 경량화해서 작은 memory 요구량과 빠른 계산시간을 가능하도록 만든 모델이다.
- GRU의 가장 큰 특징으로서, LSTM의 두 가지 종류의 vector로 존재하던 cell state vector와 hidden state vector를 일원화하여 오직 hidden state vector만이 존재한다는 것이 가장 큰 특징이다.
- 전체적인 동작 원리는 LSTM과 굉장히 유사하다.
- GRU의 hidden state vector가 LSTM의 cell state vector와 유사한 역할을 한다고 이해할 수 있다.
- 위의 식으로부터 이해할 수 있듯이, $$z_t$$가 커지면 커질수록 $$1 - z_t$$는 점점 작아진다는 것을 알 수 있다. 즉, $$h_t$$는 $$h_{t-1}$$ 과 현재 만들어진 $$\tilde h_t$$ 간의 독립적인 형태로 gate 계산을 한 후 더하는 것이 아니라, 두 정보간의 가중평균을 내는 형태로 계산된다는 것을 알 수 있다.
- 경량화된 모델임에도 자연어 처리 Application에서 LSTM에 뒤지지 않는 성능을 나타낸다.

### 3-5. Backpropagation in LSTM & GRU
![29](https://user-images.githubusercontent.com/53552847/132636614-b5d21af7-9c6d-4691-b08e-b47d0493c027.PNG)
- Uninterrupted gradient flow
- 정보를 담는 주된 벡터인 Cell state vector가 update되는 과정이 기존의 original RNN에서 동일한 $$w_{hh}$$ 행렬을 계속해서 곱해주는 형태의 연산이 아니라 전 time step의 cell state vector에서 그때그때 서로 다른 값으로 이루어진 forget gate를 곱하고 필요로 하는 정보를 덧셈을 통해 만들어준다는 사실로 인해 Gradient Vanishing 혹은 Exploding의 문제가 사라짐을 알 수 있다.
- 기본적으로 덧셈 연산은 Backpropagation을 수행할 때, Gradient를 복사해주는 연산이 된다.
- 이로 인해, Original RNN에 비해 멀리있는 time step 까지 Gradient를 큰 변형없이 전달해줄 수 있다. 이를 통해, 더 긴 time step간에 존재하는 long term dependency 문제를 해결할 수 있다.

### 3-6. Summary on RNN / LSTM / GRU
- RNN은 다양한 길이를 가질 수 있는 Sequence data에 특화된 보다 유연한 모델 구조이다.
- Original RNN은 구조가 굉장히 간단하지만, 학습 시 Gradient Vanishing, Exploding의 문제가 있어 실제로 많이 사용되지 않는다.
- LSTM, GRU를 실제로 많이 사용하고 그 방법에 대하여, LSTM에서는 Cell state vector, GRU에서는 hidden state vector를 각 time step에서 update하는 과정이 기본적으로 덧셈에 기반하는 연산이기 때문에 이러한 gradient vanishing 혹은 exploding의 문제를 피하고 long term dependency 문제를 해결할 수 있다.

## 4. 피어 세션
- 정보를 전달할 떄 tanh 를 취하는 이유는 무엇일까요?
- 틸드 C_t를 왜 제한하는 걸까요? 단순히 다음 층에 필요 없는 정보는 빼고 전달해야 하기 때문인가요?
- GRU에서 rt에 대한 자세한 설명은 skip된 것 같습니다.
- h와 c의 차이는 무엇일까?: 기억 셀(c)은 LSTM끼리만 주고 받는 정보, 은닉 상태(h)는 LSTM 바깥으로 출력 되는 output으로 일단 이해하면 될 듯.
- [RNN에서 activiation function을 tanh(x)를 쓰는 이유는 뭘까?](https://coding-yoon.tistory.com/132)  
→ 원래 있던 값을 그대로 가져가게끔 만들어져서 sigmoid를 쓰는 것보다 hyperbolic tangent를 쓰는 게 낫다고 판단했다.
→ BPTT를 쓰기 때문에 hyperbolic tangent는 미분값 range가 (0, 1) 사이이고, sigmoid는 (0, 0.25) 사이이기 때문에 이를 계속 곱해주면 sigmoid는 vanishing gradient 문제가 생긴다고 알고 있고, hyperbolic tangent는 그렇지 않다고 알고 있다. 
→ 다만 ReLU를 BPTT 구조에서 사용하면 왜 안 되는지는 모르겠다. 누군가가 `RNN에서 relu를 사용하면 vanishing gradient 문제는 해결하지만 히든레이어 값이 exploding 하기 때문에 사용하지 않는다고 들었습니다.` 라고 얘기하기는 했다.

## 5. 26일차 후기
과제의 난이도도 많이 높지 않았고, 강의 내용 역시 지난 시간에 들었던 내용을 복습하는 느낌이라서 인지 빡쎄지는 않았던 것 같다. 확실히 과제를 하다보니 느끼는 거지만 NLP의 경우 전처리의 과정이 다소 복잡하기 때문에, 코드도 길어지는 것 같다고 느꼈다.

강의의 내용을 충분히 소화했다고 느꼈음에도, 과제에서 이해가 되지 않는 부분들이 다소 많았기에 해결은 했지만, 이해할 수 있도록 복습이 필요할 것 같고 실습 코드 내용역시 알찬 구성으로 이루어져 있어서 NLP를 진행할 때 동안은 꼭꼭 실습도 내껄로 만들려고 노력할 것이다!! 

열심히해보자~~

## 6. 해야할 일
- RNN의 $$W_h$$, $$W_x$$는 모든 time step에서 동일한가?
- original RNN에서 선형결합 후 tanh를 통해서 최종 hidden state vector를 -1 ~ 1의 값으로서 어떤 유의미한 정보를 담는 역할을 한다고 하는데 다른 activation function을 사용하면 안되는가?
- GRU에 대한 심화 내용학습
- LSTM & GRU에서의 backpropagation 내용 심화 이해
- BPTT 이외에 RNN / LSTM / GRU 구조를 유지하면서 gradient vanishing 혹은 exploding 문제를 완화할 수 있는 방법이 있을까?
- RNN / LSTM / GRU 기반의 Language Model에서 초반 time step의 정보를 전달하기 어려운 점을 완화할 수 있는 방법이 있을까?
- 
