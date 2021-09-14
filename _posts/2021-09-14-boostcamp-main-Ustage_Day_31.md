---
layout: post
title: "NLP Ustage Day 7 (Day 31)"
categories: boostcamp
tags: main
comments: true
---
Transformer에 대하여 학습한다.

**부스트 캠프 31일차 학습 요약**
- **학습** : Transformer(2)
- **피어 세션**

## 목차
- [1. Transformer](#1-transformer)
- [2. 31일차 후기](#2-31일차-후기)
- [3. 해야할 일](#3-해야할-일)

## 1. Transformer
### 1-1. Multi-Head Attention
![9](https://user-images.githubusercontent.com/53552847/133261563-1483c9df-1130-4c93-ac33-7a24d4f2634c.PNG)
- Input vector를 Q, K, V vector로서 입력을 주고 이들 각각을 서로다른 선형변환을 통과시킨 후, Q, K의 내적연산을 진행하고 Softmax를 취한 뒤 V와의 weighted sum을 통해 각 word에 대한 최종 encoding vector를 생성한다.
- 이러한 과정이 위의 그림에서 h개로 중첩되어 있는 것을 볼 수 있는데, 이는 동일한 Q, K, V vector들에 대하여 동시에 병렬적으로 여러 버전의 attention을 수행한다는 의미이다.
- 구체적으로 $$W_q$$, $$W_k$$, $$W_v$$가 하나의 행렬 세트만 존재하는 것이 아니라 다음의 식처럼 여러 버전의 행렬들이 존재하고, $$i_{th}$$ 의 attention에는 해당하는 i번째의 $$W_q$$, $$W_k$$, $$W_v$$ 행렬을 사용하여 선형변환을 한 후 attention을 진행하고 이로부터 Q vector들에 대한 각각의 encoding vector를 얻게 된다.
![10](https://user-images.githubusercontent.com/53552847/133261569-47519c8d-1f69-4be1-b883-3b5b1c1457b5.PNG)
- 여러 버전의 attention을 수행하기 위한 선형변환 행렬들을 서로 다른 head라고 부르고 이를 다수의 버전의 attention을 수행한다는 의미로서 Multi-Head Attention이라고 부른다.

#### Why do we use Multi-Head Attention?!
- 어떤 동일한 sequence 내에서도 어떤 특정한 Query word에 대하여 서로 다른 기준으로 여러 측면에서의 정보를 뽑아야할 필요가 있다.
- 예를 들어, 'I went to the school', 'I studied hard', 'I came back home and I took the rest'가 있을 때, I라는 Query word에 대한 encoding을 수행하기 위해서는 I가 한 행동을 중심으로 정보를 뽑을 수도 있고, I가 존재했던 장소의 변화에 대한 정보를 뽑을 수도 있는 것이다.
- 이러한 방식으로 서로 다른 측면의 정보를 병렬적으로 뽑고 그 정보를 전부 합쳐주는 형태로 attention을 구성하고, 각각의 head가 앞서 말한 것처럼 서로 다른 정보들을 상호보완적으로 뽑는 역할을 한다.
- 이를 도식화하면 아래의 그림과 같다.
![11](https://user-images.githubusercontent.com/53552847/133261571-4983a742-e88a-406f-9e36-8f4c009596a8.PNG)
- 위 그림은 두 단어로 이루어진 sequence를 예로 그린 그림인데, 두 개의 서로 다른 head가 존재한다고 가정한 경우, 동일한 vector x에 대하여 각각의 head 별로 존재하는 선형변환 행렬을 통해서 얻어지는 Q, K, V vector를 바탕으로 각 head 내에서의 encoding vector를 얻을 수 있다.
![12](https://user-images.githubusercontent.com/53552847/133261572-5d9999ca-61d2-433f-885a-0cea06ebeacb.PNG)
- 앞에서 얻은 각 head의 encoding vector들을 위의 그림처럼 concat한 형태로 합쳐주면 (Word의 개수 * ($$d_v$$ * head 수))의 dimension을 가지게 되고, 이를 맨 처음의 특정 dimension으로 맞춰주기 위하여 Linear Transformation을 진행한다.

#### Attention Model의 계산량 및 메모리 요구량
![13](https://user-images.githubusercontent.com/53552847/133261574-7ef9282b-d0e7-4c27-9bba-234cf8257811.PNG)
- Self-Attention에서의 주된 계산 부분을 차지하는 것은 앞서 말한 Q, K의 내적부분이다. 즉, Q, K의 개수의 곱 ($$n^2$$)과 dimension을 곱한만큼의 계산량을 필요로 한다.
- 행렬연산은 GPU 코어 수가 충분히 많다면 sequence의 길이가 아무리 길고, dimension이 아무리 크더라도 이 모든 계산을 GPU가 가장 특화된 행렬 연산의 병렬화를 통해 코어 수가 무한정 많다는 전제하에 이를 모두 한 번에 계산할 수 있다.
- RNN의 경우, 매 time step마다 hidden state vector에 대한 연산을 진행해야하므로 $$d^2$$만큼의 연산을 진행하게 되고 time step의 개수인 n만큼 곱해서 위의 표처럼 $$nd^2$$만큼의 계산량을 필요로 한다.
- 메모리 요구량의 측면에서는 각 입력 문장의 길이에 따라 self-attention의 경우 제곱 단위로 늘어나기 때문에 RNN에 비해서 훨씬 많은 메모리가 필요하다. 이는 Q, K의 내적에 대한 값들을 메모리에 저장하고 있어야함으로 이해할 수 있다.
- 병렬화 측면에서, RNN, Self-Attention 모델의 차이는 앞서 말한 것처럼 self-attention의 경우 sequence가 아무리 길더라도 GPU 코어 수가 충분히 뒷받침 된다면 그 모든 계산을 동시에 수행할 수 있지만, RNN의 경우 해당 time step에서의 hidden state vector의 경우 이전 time step을 거처야만 계산될 수 있기 때문에 병렬화가 불가능해지고, RNN의 forward propagation, backward propagation은 모두 sequence 길이에 비례하는 형태로 병렬화가 불가능한 sequential한 operation이 필요하다.
- 이러한 사실들로 부터, self-attention 기반 transformer 모델의 학습은 RNN에 비해 훨씬 빨리 진행될 수 있지만, RNN보다 훨씬 많은 메모리량을 필요로 한다.
- 더불어, Maximum Path Length의 경우 Long-Term Dependency와 직접적으로 관련이 있는데, sequecne에서의 각 word 사이의 gap이 클 때, RNN은 time step만큼을 무조건 거쳐야 해당 단어에 대한 정보를 얻을 수 있지만, self attention의 경우 바로바로 얻을 수 있다. 이런 측면에서 볼 때, Maximum Path Length는 위의 표에서 처럼 RNN은 O(n)이고 self attention은 O(1)임을 알 수 있다.

### 1-2. Block-Based Model
![14](https://user-images.githubusercontent.com/53552847/133261575-e994c98e-43ac-4aca-88d0-c82832e2ae08.PNG)
- Each block has two sub-layers
    - Multi-Head Attention
    - Two-layer feed-forward NN (with ReLU)
- Each of these two steps also has
    - Residual connection and layer normalization
    - LayerNorm(x + sublayer(x))
- Multi-Head Attention 뿐만 아니라 Residual Connection을 수행하고 Layer Normalization을 수행한다.
- 추가적으로 Feed Forward Network를 통과하고 여기서도 Residual Connection 및 Layer Normalization을 수행한다.
- Feed Forward Network와 Residual Connection이 없으면 Attention Matrix가 기하급수적으로 1-rank matrix로 수렴하므로 Feed Forward Network와 Residual Connection을 사용한다. (자세한 내용은 [여기](https://arxiv.org/abs/2103.03404)에서 확인해보자.)
- 더불어, LayerNorm은 Pre-trained Language Model에서 성능에 귀결되는 핵심 재료라고 한다.(자세한 내용은 [여기](https://arxiv.org/pdf/2103.05247.pdf)에서 확인하자)
- 각 Block에서의 output은 input과 동일한 차원의, input vector 각각에 대응하는 encoding vector가 된다.

### 1-3. Residual Connection
- Computer Vision에서 깊은 Layer의 Neural Network을 만들 때, Gradient Vanishing 문제를 해결하여 학습은 안정화 시키면서 layer를 쌓아감에 따라 더 높은 성능을 낼 수 있도록 하는 효과적인 방법이다
- 기존의 Input vector와 Multi-Head Attention에서 나온 encoding vector를 add하여 encoding vector를 만들어 준다.
- 이런 과정을 통해, Gradient Vanishing의 문제를 해결할 수 있고, 학습도 조금 더 안정화 시킬 수 있다.
- Residual을 적용하기 위해서 input vector와 attention 모듈의 출력인 encoding vector의 dimension이 정확하게 일치해야 한다.

### 1-4. Layer Normalization
![15](https://user-images.githubusercontent.com/53552847/133261576-cc417913-fcd3-4366-96ee-0f1f5690bd82.PNG)
- Layer Noramlization changes input to have zero mean and unit variance, per layer and per training point (and adds two more parameters)
- Normalization은 여러가지가 존재하고 위의 그림처럼, Batch Norm, Layer Norm, Instance Norm, Group Norm 등이 존재한다.
- 기본적으로 Normalization은 주어진 다수의 sample 들에 대하여 그 값들의 평균을 0으로 분산을 1로 만들어 준 후 우리가 원하는 평균과 분산을 주입할 수 있도록 하는 선형변환으로 이루어진다.
- 기존의 평균과 분산이 얼마였던지 간에 그 정보를 버리고 표준화된 평균과 분산 즉, 0, 1로 만들어주는 과정으로 해석할 수 있으며 그 이후에 각각의 값들에 하나의 affine transformation 연산을 수행하여 평균과 분산을 우리가 원하는 형태로 바꾸어준다.
- 예를 들어, affine transformation의 형태가 y = 2x + 3의 변환을 수행할 때, 평균과 분산이 0, 1로 바뀐 값들이 x로 들어가 y로의 변환이 되는데, 이 때 평균은 3, 분산은 $$2^2$$의 형태로 변환된다.
- affine transformation에서의 2와 3(위에서) 같은 값들은 Neural Network가 최적화에 의해서 Gradient Descent에 의해 최적화를 하는 parameter가 되고 그렇게 됨으로서 Neural Network 학습 과정에서 특정 노드에서 발견되어야 하는 값의 가장 최적화된 평균과 분산을 원하는 만큼 가지도록 조절한다.  
![16](https://user-images.githubusercontent.com/53552847/133261579-73328a28-5371-426f-b6dd-d41aae549d0d.PNG)
- 위의 예시에서 볼 수 있듯이, 'thinking machines'라는 문장이 각각 4차원 벡터로 표현되어 있다고 할 때, 이를 각 word 별로 평균과 분산이 0, 1이 될 수 있도록 진행하고, 우리가 원하는 평균과 분산을 주입하기 위해 affine transformation을 수행하며 이는 각 노드별로 여러 단어에 걸쳐 공통적인 변환을 적용해주게 된다.
- Batch Normalization과 세부적인 차이점은 있지만, 큰 틀에서는 학습을 안정화하고 최종적인 성능을 끌어올리는데 중요한 역할을 하는 것은 비슷하다.

### 1-5. Positional Encoding
#### self-attention의 순서 정보
- RNN과 달리, self-attention을 기반으로 주어진 sequence를 encoding하는 경우 이전에 설명한 과정을 거쳐 각 word별 encoding vector가 출력되는데, 이때 순서의 측면에서 input word의 순서를 뒤집더라도 각 word의 encoding vector는 동일하게 나온다.
- 더불어, attention 모듈을 수행할 때, K, V pair들은 순서에 상관없이 Q별로 K들과의 유사도를 구하고 이로부터 V에 weighted sum을 하여 encoding vector를 얻는데 weighted sum을 구할 때 V vector들의 교환법칙이 성립하여 output vector는 순서에 상관없이 동일하게 출력된다.
- 위처럼, 순서를 무시한다는 특성으로부터 입력 문장을 어떤 sequence 정보를 바탕으로 encoding하지 못하고, 마치 순서를 고려하지 않는 set의 형태로 보고 이로부터 encoding vector를 얻는 과정으로 이해할 수 있다.
- RNN의 경우, sequence word들의 순서가 달라지게 되면 hidden state vector에 적층되는 단어가 달라지기 때문에 encoding vector가 당연히 달라지게 되지만 self-attention은 순서정보를 반영할 수 없다는 한계점이 있다.
- 이로부터 Positional Encoding이 생겨났다.

#### Positional Encoding
- 각 word에 대한 input vector에 대하여 현재 어디에 위치하고 있는 지에 대한 정보를 함께 넣어준다.
- 가장 직관적인 예로서, 각 위치에 해당하는 element에 1000을 더해줘서 이 단어가 어디에 위치했는지를 드러낼 수 있게 한다. 즉, 기존의 input vector가 (3, -2, 4)일 때, 첫번째 위치 word라면 (1003, -2, 4)로, 세번째 위치 word라면 (3, -2, 1004)로 표현하도록 하는 것이다.
- 핵심 아이디어는 각 word의 순서를 규정할 수 있는 혹은 순서를 특정지을 수 있는 unique한 상수 벡터를 word input vector에 더해주는 것이다.
- 더해주는 vector를 어떻게 결정하는가?!
    - 앞서 말한 1000은 정말 간단한 예시이며, 실제로 이런 방식을 사용하지는 않는다.
    - Transformer 논문에서는 sin, cos로 이루어진 주기함수를 사용하고, 그 주기를 서로 다른 주기를 사용하여 여러 sin, cos 함수를 만들고 거기서 발생된 특정 x값에서의 함수값을 모아 위치를 나타내는 vector로 사용하게 된다.
![17](https://user-images.githubusercontent.com/53552847/133261580-e4c5261e-8578-4cbb-8a16-ef5765921609.PNG)
    - 보통 위의 식에서처럼 각 dimension 별로 sin, cos 함수를 만드는 데, 위의 그림으로 부터 알 수 있다시피, dimension 개수 만큼 서로 다른 주기, 그리고 sin, cos를 번갈아가며 그래프 생성 패턴을 만들어 놓은 후, 각 위치 혹은 position을 나타내는 그래프로부터 값을 얻을 수 있다.
    - 위의 방식에 따라, 순서를 구분하지 못하는 self-attention 모듈의 한계점을 위치별로 서로 다른 vector를 더해지도록 함으로서 위치가 달라지면 출력 encoding vector도 달라지게 하는 방식으로 순서라는 정보를 다룰 수 있도록 하였다.

### 1-6. Warm-up Learning Rate Scheduler
![18](https://user-images.githubusercontent.com/53552847/133261583-3fb9305e-b357-4d48-a4f1-5f7a5cd23c3b.PNG)
- Learning Rate Scheduling은 일반적으로 Gradient Descent 및 그에 대한 다양하고 좀 더 진보된 Algorithm인 adam등의 과정을 통해 최적화를 수행하는 과정에서 사용되는 Learning Rate를 적절히 변경해서 사용하는 것이다.
- Learning Rate는 학습 전 과정동안 하나의 고정된 값으로서 사용하게 되는데 이에 대해 학습을 좀 더 빠르고 최종 수렴 모델의 성능을 더 올릴 수 있도록하는 목적으로서 Learning Rate 값도 학습 중에 적절히 변경해서 사용하게되는 것이 Learning Rate scheduling 방식이다.
- 흔히 초반에는 Learning rate를 아주 작은 값을 사용해서 원래의 해당 gradient의 절대값이 너무 큰 경우 적절히 작은 learning rate를 곱해 너무 큰 보폭이 발생하는 것을 방지하고 어느 정도 완만한 구간에 도달했을 때, local optima에 빠지는 것을 방지하기 위해 동력을 좀 더 많이 주는 차원에서 learning rate를 더 키워준다.
- 위와 같은 방식의 Learning rate scheduler를 사용했을 때, 경험적으로 transformer 모델이 여러 task에서 좀 더 좋은 성능을 내는 것으로 알려져 있다.

### 1-7. 전반적인 형태
![19](https://user-images.githubusercontent.com/53552847/133261584-a46764e8-54bc-4fb7-9463-e8c469df40f3.PNG)
- 각 word별로의 Embedding vector를 입력으로 준다.
![20](https://user-images.githubusercontent.com/53552847/133261588-ba395312-9530-4ed1-acdf-a32591d36370.PNG)
![21](https://user-images.githubusercontent.com/53552847/133261590-2dfa8ab8-0af6-4b7f-a653-39135f20610a.PNG)
![22](https://user-images.githubusercontent.com/53552847/133261594-c282844d-ef7f-4bc0-88aa-65935fff78f1.PNG)
![23](https://user-images.githubusercontent.com/53552847/133261598-c8e9307b-a407-4f15-9865-81e4dada8032.PNG)
## 3. 해야할 일
- 어떻게 Multi-Head Attention에서 각 head마다 다른 정보들이 뽑혀져 오는가?
- head가 앞서 말한 것처럼 서로 다른 정보들을 상호보완적으로 뽑는 역할을 한다.라는데 어떻게 서로 다른 정보들을 뽑는 거지?!
- 우리가 원하는 평균과 분산을 주입하기 위해 affine transformation을 수행하며 이는 각 노드별로 여러 단어에 걸쳐 공통적인 변환을 적용해주게 된다.라는데 이것은 왜 단어별로 평균 분산을 0, 1로 만들어주고 노드별로 affine transformation을 진행하는가?
