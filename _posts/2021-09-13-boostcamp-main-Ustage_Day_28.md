---
layout: post
title: "NLP Ustage Day 6 (Day 30)"
categories: boostcamp
tags: main
comments: true
use_math: true
---
Transformer에 대하여 학습한다.

**부스트 캠프 30일차 학습 요약**
- **학습** : Transformer(1), bahdanau attention 정리

## 목차
- [1. Transformer](#1-transformer)
- [2. 30일차 후기](#2-30일차-후기)

## 1. Transformer
- Self Attention은 RNN 기반 Translation Model의 단점을 해결하기 위해 처음 등장했다.
- RNN & Attention을 함께 사용했던 기존과는 달리 Attention 연산만을 이용하여 입력 문장의 representation을 학습하여 좀 더 parallel한 연산이 가능함과 동시에 학습속도가 빠르다는 장점을 보였다.

### 1-1. Transformer: High-level view
- Attention is all you need, NeurIPS'17
- No more RNN or CNN modules

### 1-2. RNN: Long-term dependency
![1](https://user-images.githubusercontent.com/53552847/133054473-c5824a2a-cb42-4a1e-ac2f-c8cf8a1e5b2c.PNG)
- 위의 그림과 같은 구조를 바탕으로 각 word의 정보가 흐른다.
- time step이 지나면 지날수록 처음에 들어왔던 word에 대한 정보는 유실 혹은 변질될 수밖에 없다.

### 1-3. Bi-Directional RNNs
![2](https://user-images.githubusercontent.com/53552847/133054485-7419a2cd-7250-401f-9fe8-ddb297c2ddfb.PNG)
-  왼쪽에서 오는 정보 뿐만 아니라 오른쪽에서 오는 정보 역시 함께 포함할 수 있도록, forward RNN과 backward RNN 두 개의 모듈을 병렬적으로 만든다.
-  특정 time step t에 대하여 forward RNN에서 만들어진 hidden state vector와 backward state vector를 concatenate하여 기존의 hidden state vector의 2배에 해당하는 벡터를 각 단어의 encoding vector로서 가지게 된다.

### 1-4. Transformer: Long-Term Dependency
![3](https://user-images.githubusercontent.com/53552847/133054486-a80cba34-3464-413e-8eb9-0841b7f70709.PNG)
- 이전에 설명했던 모듈들과 동일하게 입력 word에 대한 input vector가 주어질 때, 출력으로 나오는 vector들은 input sequence에서 주어진 각 단어별로 sequence 전체 내용을 잘 반영한 encoding vector가 output으로 나오는 것으로서, 입출력의 세팅은 동일하게 유지된다.
- Seq2seq with Attention 모듈에서, decoder에서의 특정 time step에 대한 hidden state vector를 Transformer에서는 encoder에서의 hidden state vector라고 볼 수 있다.
- 각 단어들의 hidden state vector의 내적을 바탕으로 내적에 기반한 유사도를 구하게 되는데, 이로부터 구한 유사도를 softmax를 취하여 가중치를 구한다. 이러한 가중치를 입력 hidden state vector과 연산하여 이를 가중평균하여 입력 word에 대한 encoding vector를 구할 수 있다.
- 이전에 설명했던 모듈들에서는 decoder hidden state vector와 input hidden state vector 사이의 유사도를 구하였는데, 본 Transformer에서는 encoder hidden state vector들 사이의 유사도를 구한다는 측면에서 self-attention 모듈이라고 한다.

### 1-5. 확장된 형태의 attention
- self-attention을 통해 자기 자신과 내적 연산을 하게될 경우, 서로 다른 벡터와 내적 연산을 했을 때보다 훨씬 더 큰 값이 추출되고 결국 자기 자신에게 큰 가중치가 걸리는 형태가 나오게 되고 이로써 자기 자신의 정보만을 주로 포함한 벡터가 나오게 된다. 이러한 issue를 해결하기 위하여 확장된 형태의 attention을 사용한다.
- 주어진 입력 벡터에서의 각 word에 해당하는 벡터들은 각각의 상황마다 다른 역할을 한다. (내적 연산을 통해 유사도를 구하고, 가중치합을 구한다.)
- 먼저, 입력 vector가 마치 이전에 설명했던 decoder hidden state vector인 것처럼 우리에게 주어진 벡터 set들의 가중치, 유사도를 구하는 형태의 벡터로 사용되는데, 이 때 어떤 벡터를 선별적으로 가져올지에 대한 기준이 되는 벹어가 Query이다.
- 어떤 단어의 Query 벡터가 각각의 단어들의 hidden state vector들과 내적을 통해 유사도가 구해지는데, 이 때 내적이 되는 각각의 재료 vector들을 Key vector라고 한다. 즉, 주어진 여러개의 key vector들 중에서 Qeury vector와 어떤 key vector가 높은 유사도를 가지는지에 대하여 찾을 수 있다.
- 유사도를 softmax 취하여 이를 가중치로서 사용한다. 이러한 가중치와 실제로 연산되는 input hidden state vector를 value vector라고 하며, 이 연산을 통해 최종 가중평균된 encoding vector를 추출한다. 이 value vector는 처음에는 실제로 key vector와 동일하지만 쓰임이 다르다.
- 하나의 sequence를 Attention을 통해 encoding하는 과정에서, 각 hidden state vector 들이 Q, K, V로서의 세 가지 역할을 하고 있는 것을 볼 수 있으며, 이들은 동일한 vector에서 출발했다 하더라도 각 역할에 따라 서로 다른 형태로 변환할 수 있도록, 별도의 linear transformation matrix가 각각 따로 정의 되어있어 3가지 역할이 모두 공유 되는 것이 아니라 서로 다른 vector로 변환될 수 있는 형태가 만들어진다.
- K, V vector는 Q가 K에 의해 유사도를 연산한 후 이를 매칭되는 V와의 연산을 진행해야 하므로, K와 V의 개수는 정확하게 동일해야한다.
- 1-4의 그림에서 볼 수 있듯이, 위의 방식대로 진행하게 되면, 첫 번째 단어로부터 Q, K, V vector를 만들었다 하더라도, 자기 자신과의 내적값이 다른 단어와의 내적보다 작을 수도 있게 된다. 이는, Q, K로의 서로 다른 변환이 존재하고 이로 인해 같은 벡터를 가지고 Q,  K를 만들었을 때에도 좀 더 유연한 유사도를 얻을 수 있다.
- Q, K로부터 가중치가 얻어지고 이를 V와 연산하여 가중평균을 냄으로서 최종적인 각 word에 대한 encoding vector가 출력된다.
- Self-Attention 모듈에서는 sequence의 길이가 길다 하더라도, 각 단어들은 Q, K, V 각각의 hidden state vector가 만들어질 것이고, Q, K 연산으로부터의 내적값이 크기만 하다면 즉, 유사도가 높기만 한다면 time step에 무관하여 멀리 있는 정보를 손쉽게 가져올 수 있다.
- 이로서, 기존의 RNN에서 보이던 한계점을 극복할 수 있게 된다. 즉, Transformer Model은 근본적으로 Long-Term Dependency의 문제를 깔끔하게 해결한 형태의 sequence encoding 기법으로 생각할 수 있다.

### 1-6. Transformer: Scaled Dot-Product Attention
- Inputs: a query q and a set of key-value (k, v) pairs to an output
- Qeury, Key, Value and Output is all vectors
- Output is weighted sum of values
- Weight of each value is computed by an inner product of query and correspoding key
- Queries and Keys have same dimensionality $$d_k$$ and dimensionality of Value is $$d_v$$
- Q와 K는 내적 연산이 가능해야하기 때문에, 같은 dimension이어야 하며, V는 softmax연산을 통해 계산된 scalar형태의 가중치와 연산이 되기 때문에 Q, K와 dimension이 동일할 필요는 없다.
![4](https://user-images.githubusercontent.com/53552847/133054487-94460a77-300d-4035-828c-ae647d66ccbb.PNG)
- Attention module의 최종적인 output의 차원은 "word의 개수 x $$d_v$$"가 된다. 흔히 output의 차원은 $$d_v$$라고 얘기하며 "word의 개수 x $$d_v$$"라고 말할 수 있는 이유는 실제 Transformer 구현상으로 동일한 shape으로 mapping된 Q, K, V가 사용되어 각 matrix의 shape은 모두 동일하기 때문이다.
- 위의 query vector q에 대하여 stack한 형태의 matrix Q로 나타내면 다음의 그림처럼 나타낼 수 있다. 
![5](https://user-images.githubusercontent.com/53552847/133054488-5021252e-709d-4e0e-ac74-42895e777412.PNG)
![6](https://user-images.githubusercontent.com/53552847/133054490-6e024093-4858-4b8a-a31e-af5722553b41.PNG)
- 특히, softmax 연산의 경우, 각 query 별로 연산을 수행해야 하므로 row-wise하게 연산한다.
- Query vector에 대한 attention model의 계산을 행렬 연산으로 바꾸어 진행하게되면 GPU를 활용하여 굉장히 빠르게 병렬화하여 계산할 수 있으며 이러한 병렬적 행렬 연산을 바탕으로 transformer 모델은 기존의 RNN등에 비해서 상대적으로 학습이 빠른 특성을 가진다.(질문 후 삭제 :44:10)
![7](https://user-images.githubusercontent.com/53552847/133054493-aa787175-a23f-4f2c-b154-692238a44c60.PNG)
- softmax를 취하기 전에 Q, K의 내적값에 대하여 sqrt($$d_k$$)로 나눠준다.

![8](https://user-images.githubusercontent.com/53552847/133054495-b63211ad-13db-4fd3-ab45-f6ffb5f20478.PNG)
- Scaling 과정으로서 sqrt($$d_k$$)로 내적값을 왜 나눠주는가?
    - Q, K의 원소들이 통계적으로 서로 독립이고 평균이 0, 분산이 1인 확률변수라고 가정할 때, 내적을 한 후의 평균과 분산은 각각 0, 'element의 수 * 분산'이 된다.
    - 즉, Q, K의 dimension에 따라, 분산이 커지는 것을 알 수 있고, Q, K vector의 dimension이 엄청나게 커졌을 경우 이로 부터 연산된 유사도를 softmax 취하게 되면 확률분포가 어떤 큰 값에 몰리는 형태의 패턴이 나타날 수 있다.
    - 반대로, 분산이 작았을 경우에는 확률 분포가 좀 더 고르게 Q에 대한 K의 확률값이 uniform distribution에 가까운 형태로 나타날 수 있다.
    - 이러한 이유로 인하여, 내적의 참여하는 Q, K의 dimension이 얼마냐에 따라서 내적값의 분산이 크게 좌지우지 될 수 있고 이에 따라 의도치 않게 softmax로 부터 나온 확률분포가 어느 하나의 key에만 몰리는 극단적인 확률로 나올 수 있다.
    - 이렇기에, Q, K의 내적값을 일정하게 유지시켜줌으로서 학습을 안정화시킬 수 있는데 이 때 내적값을 sqrt($$d_k$$)로 나눠주어 일정하게 유지시킬 수 있다.
    - 이렇게 되면, Q, K의 dimensino이 얼마였던지 간에 상관없이 분산이 일정하게 1인 형태로 유지되는 형태로 나온다.
- softmax의 값이 한 쪽으로 굉장히 몰리게 되는 경우에, 실제로 이 상태로 학습을 진행하게 되면 gradient vanishing이 발생할 수 있는 위험성이 있고, 의도치 않게 Q, K의 dimension을 크게 설정하고 scaling 없이 attention 모듈을 수행하였을 때, 종종 학습이 전혀 진행되지 않는 형태가 발생할 수 있다.

## 5. 30일차 후기
강의의 난이도가 높아지긴 했지만 양의 그닷 많지는 않아서 나름 쉬어갈 수 있던 하루였던 것 같다. 당연하게도 강의 뿐만 아니라 공부해야할 것들, 찾아보아야 할 것들이 수도 없이 많아서 쉴틈이 없었지만 마음의 짐은 많이 없는 채로 공부할 수 있었던 것 같다!

더불어, 9주부터 20주까지 함께할 팀을 구해서 다행이나마 안심할 수 있었다. 팀원분들과 인사를 나눴는데 아니나 다를까 엄청 어색했었지만 그래도 함께할 팀원이 있다는 사실에 안도감을 느낄 수 있었고 어떤 프로젝트건 함께 열심히 해보고 싶다는 생각이 들었다.

일단 NLP 이론에 집중해서 U-stage 동안 열심히 공부하자~! 다들 파이팅!!
