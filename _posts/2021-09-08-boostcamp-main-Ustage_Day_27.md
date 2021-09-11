---
layout: post
title: "NLP Ustage Day 3 (Day 27)"
categories: boostcamp
tags: main
comments: true
---
Attention, Beam search에 대하여 학습하고 BLEU Metric에 대하여 알아본다.

**부스트 캠프 27일차 학습 요약**
- **학습** :Sequence to Seuqence with Attention, Beam Search, BLEU score
- **피어 세션**

## 목차
- [1. Sequence to Sequence with Attention](#1-sequence-to-sequence-with-attention)
- [2. Beam Search](#2-beam-search)
- [3. BLEU score](#3-bleu-score)
- [4. 피어 세션](#4-피어-세션)
- [5. 27일차 후기](#5-27일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Sequence to Sequence with Attention
### 1-1. Sequence to Sequence Model
- many to many의 형태이다.
- It takes a sequence of words as input and gives a sequence of words as output.
- It composed of an encoder and a decoder
![30](https://user-images.githubusercontent.com/53552847/132700123-7d517127-707f-463d-b2bf-ea5297281c86.PNG)
- 위의 그림은 'Are you free tomorrow?'라는 문장이 입력으로 들어왔을 때, 'Yes, what's up?'이라는 단어가 출력되는 구조를 보이고 있다.
- 이 때, 입력 문장을 읽어들이는 RNN 모델을 encoder라고 하고, 출력 문장을 순차적으로 단어 하나하나씩 생성하는 RNN model을 decoder라고 한다.
- Encoder, Decoder는 서로 share 하지는 않는 parameter로 이루어진 rnn 모듈이다.
- Encoder, Decoder의 각 cell에서는 LSTM을 선택하여 사용한다.
- Encoder의 마지막 time step에서의 hidden state vector는 decoder rnn의 $$h_0$$ 즉, 첫번째 time step의 입력으로 주어지는 이전 time step의 hidden state vector로서의 역할을 한다.
- $$h_0$$ 즉, encoder의 마지막 time step에서의 hidden state vector는 encoder에서 수집한 정보들을 잘 보관하고 있다가, 이런 정보를 바탕으로 decoder가 순차적으로 그에 대응하는 단어들을 예측한다.
- word 별로 문장을 생성하는 task에서 보통 첫번째 단어로서 넣어주는 문자는 어떤 하나의 특수 문자로서 start token 혹은 <sos> token이라고 부른다.
- 이러한 특수한 word를 vocabulary 상에 정의해두고 이를 가장 처음에 decoder time step에 넣어줌으로서 실질적으로 생성되는 어떤 첫 번째 단어부터 예측을 수행하게 된다.
- decoder에서 문장이 끝나는 시점은 언제까지나 단어별 생성을 수행할지에 대하여 정해야 하는데 이 때, 마지막에 또 다른 특수 문자인 <eos> token이 나올 때 까지 decoder rnn을 구동하게 된다.
- 이 <eos> token이 생성되면, 이를 최종출력으로 해서 더 이상 단어를 생성하지 않고 종료하는 형태의 과정을 따르게 된다.

### 1-2. Seq2Seq Model with Attention
- Attention provides a solution to the bottleneck problem
- Core Idea : At each time step of the decoder, focus on a particular part of the source sequence
![31](https://user-images.githubusercontent.com/53552847/132700127-faa12323-18b5-4639-a3cd-17dae256fed2.PNG)

#### Attention 모듈의 motive
- Bottleneck Problem
    - Seq2Seq Model에서 사용하는 RNN 기반의 Model 구조가 Seq의 앞에서부터의 정보를 순차적으로 읽어들이며 이에 필요한 정보를 매 time step마다 축적해가며 hidden state vector를 생성하는 과정을 따른다.
    - 이러한 과정에서 hidden state vector가 RNN의 구조 특성상 hidden state vector의 dimension이 항상 정해져 있다는 특성이 있고, 이로 인해 입력 문장이 짧든 길든 마지막 time step에 있는 hidden state vector에 앞서 나온 많은 정보들을 우겨넣게 되는 현상이 발생한다.
- Long Term dependency
    - LSTM을 활용하여 어느 정도의 long term dependency를 해결했다 하더라도 여전히 마지막 time step을 기준으로 멀리 있는 정보는 LSTM 구조를 지나가면서 그 정보가 점차 변질되거나 소실될 수 있다.
    - 예를 들어, 'I go home'을 한국어로 번역한다고 했을 때, 일반적으로 먼저 나오는 I를 먼저 번역해야하는데 encoder의 마지막 time step에서 나오는 hidden state vector에서는 정작 처음에 나오는 단어에 대한 정보가 유실됬을 확률이 높아 첫 단어부터 제대로 생성하지 못해 전체 품질이 낮아지는 현상이 발생한다.
    - 이러한 문제를 해결하기 위해 'I go home' 대신 'home go I'를 입력해서 마지막 time step에 문장의 초반 정보가 잘 담겨 마지막 hidden state vector가 decoder로 넘어갔을 때, 초반에 해당하는 단어들이 잘 생성될 수 있또록 하는 technique도 제안된 바가 있다.

#### Attention 모듈의 Problem Setting
- decoder는 encoder의 마지막 hidden state vector에만 의존하는 것이 아니라 각 cell에서 나온 hidden state vector 전부를 decoder에 제공해준다.
- decoder에서는 각 time step에서 단어를 생성할 때, 그때그때 필요한 encoder의 hidden state vector를 선별적으로 가져와서 예측에 도움을 주는 형태로 활용하게 된다.

#### Attention 모듈의 동작 과정
![32](https://user-images.githubusercontent.com/53552847/132700128-ed47f56a-3444-473e-b4bf-7102cbdc7304.PNG)
- 마지막 $$h_0$$와 <sos> token을 가지고 $$h^d_1$$ (decoder의 첫번째 hidden state vector)를 출력한다.
- decoder에서의 첫 번째 time step에서의 hidden state vector를 가지고 다음 단어 예측에 사용할 뿐만 아니라 encoder에서 주어진 여러개의 hidden state vector 중 현재에 어떤 hidden state vector를 필요로 하는지를 선정하는 역할을 수행한다.
- 위의 과정은 구체적으로, decoder hidden state vector가 encoder hidden state vector 각각과 내적의 연산을 수행하고 이 내적값을 decoder hidden state vector와 encoder hidden state vector 사이의 유사도로 이해할 수 있고, 이 값들을 softmax 취해서 각각의 내적값을 logit vector로 생각하여 각각의 단어에 대한 확률값으로 계산할 수 있다.
- 이렇게 나온 확률값은 각각의 encoder hidden state vector에 부여되는 가중치로서 사용이 되고, 이를 가중평균 계산하여 하나의 encoding vector를 구할 수 있다.
- 이 encoding vector를 Attention 모듈의 output으로서 context vector라고 부른다.
- Attention 모듈의 입력은 하나의 decoder hidden state vector와 encoder hidden state vector set가 들어간다.
- Attention 모듈의 출력으로서 하나의 context vector가 나오게되고, 이는 decoder hidden state vector와 concatenate되어 output layer의 입력으로 들어가게 되고 그로부터 다음에 나올 단어를 예측하게된다.

![33](https://user-images.githubusercontent.com/53552847/132700130-84e25794-2b8f-4837-99b0-c81d8f106205.PNG)
- 각 time step에서 이전의 과정을 반복적으로 수행함으로서 그때그때 서로 다른 가중치를 encoder hidden state vector에 적용해서 나오는 가중평균된 vector를 output layer에 직접적인 입력의 일부로서 사용해서 해당 time step에서의 예측 정확도를 올리는 방식으로 attention 모듈이 seq2seq 모델의 성능을 올려줄 수 있게 된다.
- decoder hidden state vector는 output layer의 입력으로 사용됨과 동시에 입력 sequence에서 주어지는 각 word별로의 encoding vector 중 어떤 단어 vector를 중점적으로 가져와야 할 지, attention 가중치를 결정해주는 역할을 수행하게된다.
- Backpropagation의 경우, output으로부터 decoder hidden state vector를 거쳐가며, context vector를 거쳐 encoder까지 가는 것으로 진행된다.
- Attention score에서 decoder rnn으로 backpropagation이 진행됨에 있어, 가중치를 선정하는 데 있어서도 학습이 진행되어 올바른 가중치를 선정할 수 있도록 학습이 진행된다.
- 학습할 때의 decoder에서의 input은 ground truth의 단어를 사용하게 되는데, 이 때 이전의 time step의 잘못된 출력 결과로부터 다음 time step의 결과 역시 잘못되게 만들 수 있는 현상을 막을 수 있다. 이런 방식을 teacher forcing 방식이라고 한다.
- 이와 반대로, 훈련 중에 실제 예측값을 다음 time step의 input으로 넣어주는 방식은 teacher forcing이 아닌 방식이 되는데, 이런 방식이 모델을 실제로 사용했을 때의 상황과 더 가깝다는 것을 알 수 있다.
- teacher forcing과 teacher forcing이 아닌 방식을 적절히 결합한 학습 방식도 있는데 다음과 같다.
    - 처음에는 teacher forcing을 이용해 모든 batch를 구성하여 학습을 진행한다.
    - 어느 정도 예측 성능이 나오면 학습의 후반부에서는 teacher forcing을 사용하지 않고 학습을 진행한다.
    - 이렇게 함으로서 이 모델이 실제로 사용되었을 때에 좀 더 부합할 수 있도록 학습을 진행할 수 있다.
- Inference에서의 decoder의 input은 이전 time step의 출력 값이 다음 time step의 input으로 들어가게 된다. 

### 1-3. Different Attention Mechanisms
![34](https://user-images.githubusercontent.com/53552847/132700132-d90db25d-a282-4611-aeb2-d34151a3b983.PNG)
- 특정 decoder의 hidden state vector를 가지고 encoder의 hidden state vector와의 유사도를 구하는 방식이 결국 내적을 통해 계산을 했는데 이 방법에 있어서 다양하게 확장 혹은 변형할 수 있다.
- 위의 식에서 $$h_t$$가 decoder hidden state vector이고, $$\bar h_s$$ 가 encoder의 각 word별 hidden state vector라고 했을 때, 위처럼, 내적, 내적을 확작한 형태의 generalized dot product, concat 기반의 attention을 할 수 있다.
- Generalized dot product
    - 기존의 내적을 활용한 두 행렬의 곱에 대하여 생각해볼 때, 다음과 같이 각 dimension에 알맞은 단위행렬을 사이에 곱해주면 동일 값으로 도출되는 것을 알 수 있고 이로부터 각 행렬 사이의 들어오는 행렬을 조금씩 수정한 형태를 general한 방법으로 이해할 수 있다. 
![35](https://user-images.githubusercontent.com/53552847/132700107-76c33668-1e1a-49d7-9fb9-0f084cd54e7b.PNG)
    - 위의 식에서 볼 때, 이렇게 삽입된 행렬은 각 원소에 가중치를 주는 것으로 이해할 수 있는데, 대각원소의 경우 각 dimension 별로의 가중치, 비대각원소의 경우 서로 다른 dimension 별로의 가중치로 이해할 수 있다.
    - 이처럼 단순히 내적을 이용해 유사도를 구하는 방식에서 그 사이에 학습가능한 형태의 행렬을 둠으로서 단순한 내적의 형태보다 확장한 형태로 모델을 구성하여 attention의 유사도를 구할 수 있고 이를 보다 일반화된 dot product 형태로 generalized 유사도라고 한다.
- Concat 기반 Attention
    - 입력으로 주어지는 decoder hidden state vector, 이와 유사도를 구해야하는 encoder의 특정 word의 hidden state vector가 입력으로 주어졌을 때, 이 벡터간의 유사도를 구해야하는데, 이 유사도 값을 최종 scalar 값으로 한 MLP로서 학습가능한 형태의 Neural Network를 만들 수 있다.
    - 각 벡터들을 모두 입력벡터로 두고 fully connected layer를 지나 최종적으로 scalar값을 출력하여 이를 유사도로 활용하는 모델을 구상할 수 있다.
    - 뿐만 아니라 다음과 같이 layer를 더 쌓아서 유사도를 구할 수 있다.
![35-1](https://user-images.githubusercontent.com/53552847/132778213-4c41c03c-7df3-4163-9331-49524a99a58c.PNG)
    - 즉, 두 벡터가 concatenate되어 입력으로 들어가기 때문에 이를 concat 기반의 attention 방식이라고 한다.
    - layer를 쌓았을 때, 마지막 layer는 최종 출력을 scalar로 만들어주어야 하기 때문에 마지막 Parameter W는 행렬의 형태가 아닌 벡터의 형태를 띄게 된다.
- 유사도에 사용하는 Parameter의 최적화는 어떻게 이루어지는가?
    - 유사도에 학습가능한 형태의 Parameter를 사용함으로서 기존의 내적만으로 구성된 attention score 부분이 trainable하게 바뀐다.
    - 기존의 Backpropagation을 통해서 attention score 부분의 Parameter 역시 학습이 진행된다는 것을 알 수 있다.
    - 즉, end to end learning에 의해 유사도에서 사용하는 Parameter 역시 학습이 진행된다는 것을 알 수 있다.

### 1-4. Attention의 장점 및 특성
- Attention significantly improves NMT performance
    - It is useful to allow the decoder to focus on particular parts of the source
- Attention solves the bottleneck problem
    - Attention allows the decoder to look directly at source; bypass the bottleneck
- Attention helps with vanishing gradient problem
    - Provides a shortcut to far-away states
- Attention provieds some interpretability
    - By inspecting attention distribution, we can see what the decoder whas focusing on
    - The Network just learned alignment by itself
    
## 2. Beam Search
### 2-1. Greedy decoding
- 매 time step마다 가장 높은 확률을 가지는 단어 하나만을 택해서 decoding을 진행하게 되는데 이를 greedy decoding이라고 한다.
- 즉, Seqeunce로서의 전체적인 문장의 어떤 확률값을 보는게 아니라 근시안적으로 현재 time step에서 가장 좋아보이는 단어를 선택하는 형태의 approach이다.
- 이 경우, 단어를 잘못생성한 경우에 이후 단어 생성에 악영향을 끼칠 수 있고, 단어가 잘못 생성됨을 알았다 하더라도 이전으로 돌아가서 수정할 수 없다는 단점이 있다.

### 2-2. Exhaustive search
- Ideally, we want to find a (length T) translation y that maximizes
![36](https://user-images.githubusercontent.com/53552847/132700111-ca223f48-e483-41f5-9ca0-92d01ed1adde.PNG)
    -  위의 식에서 알 수 있듯이, 첫번째 word를 예측하는 것이 가장 높은 확률로 예측되더라도 뒤의 확률값들이 낮게 나온다면 전체적인 값은 작아질 수 있다.
    -  즉, 뒤의 확률값들이 보다 높은 확률값을 가질 수 있도록 첫번째 확률값부터 조정해나가야 전체적인 확률을 높일 수 있다.
- we could try computing all possible sequences y
    - This means that on each step t of the decoder, we are tracking $$V^t$$ possible partial translations, where V is the vocabulary size
    - 위와 같이, 각 단어를 추출할 때마다 vocabulary에 속해있는 단어를 모두 고려하게 되면 시간이 vocabulary 사이즈 만큼 걸리고 각 time step마다 지나게 되면 time step이 지날 때마다 시간은 기하급수적으로 증가하게 된다.

### 2-3. Beam Search
- Exhaustive search 방법에서의 차선책으로서 나온 방법이 Beam Search이다.
- 가장 높은 확률값을 가지는 greedy decoding과 가능한 모든 방법을 전부 확인하는 Exhaustive search의 중간쯤에 있는 Approach이다.
- Core idea : on each time step of the decoder, we keep track of the k most probable partial translations (which we call hypothesis)
    - k is the beam size (in practice around 5 to 10)
    - 우리가 정해놓은 가능한 k개의 가지수를 고려한다.
    - k개의 candidate 중에서 가장 확률이 높은 것을 선택한다.
- A hypothesis $$y_1$$, ..., $$y_t$$ has a score of its log probability 
![37](https://user-images.githubusercontent.com/53552847/132700113-85b748e5-95f5-4053-a8fc-8d2cb8d72062.PNG)
    - log는 단조증가 함수이므로, 확률값이 가장 큰 값일 때, 로그를 취해도 가장 큰 값을 유지함을 알 수 있다.
    - 확률값은 분수의 형태로 이루어져 있고, 이를 로그 취하면 Score들은 모두 음수를 가지고 가장 높은 값을 가지고 있는 것이 더 좋다.
    - We search for high-scoring hyptheses, tracking the top k ones on each step
- Beam search is not guaranted to find a globally optimal solution
- But it is much more efficient than exhaustive search

### 2-4. Beam search Example
- Start로부터 시작하고, vocabulary 상의 단어들 중에서 확률분포로 부터 가장 높은 확률값을 가지는 k(beam size)개의 word를 선택한다.
- 이런 과정을 각 k개에 대하여 올 수 있는 단어를 다시 k개 선택하고 이중에서 최종 확률값을 바탕으로 가장 높은 확률을 가지는 k개를 선택하여 반복한다. 즉, 다음의 그림과 같이 이해할 수 있다.
![38](https://user-images.githubusercontent.com/53552847/132700115-8f3e766c-34a4-4256-8aa7-aa685460af7b.PNG)

### 2-5. Stop criterion
#### Greddy decoding
- <eos>를 예측 단어로 예측했을 때, 끝난다.

#### Beam Search
- 서로 다른 hypotheses가 존재하고 이들은 각각 다른 시점에서 <eos>을 생성한다.
- 어떤 hypothesis가 <eos>를 생성했다면, 그 경로에 대해서는 완료된 것으로 하고 이를 저장공간에 임시로 저장한다.
- 남은 hypothesis에서는 decoding을 계속해서 수행한다.
- 중단
    - 우리가 정한 특정 최대 time step t를 정했을 때, t까지만 decoding을 하여 중단할 수 있다.
    - 우리가 임시 저장공간에 저장해둔 end token을 명시적으로 발생하여 완료된 hypotheses가 미리 정해준 n개 만큼 생성되면 beam search를 중단할 수 있다.

### 2-6. Finishing up
- 최종적으로 completed hypotheses 리스트를 얻게 된다.
- 이 중에서 가장 높은 확률값을 가지는 하나의 값을 선택한다.
- 완성된 sequence의 길이가 서로 다를 경우에 상대적으로 짧은 sequence일 경우 joint probability 값이 더 높기 때문에, 이런 경우에 대하여 공평하게 비교하기 위해 각 hypotheses가 가지고 있는 단어의 개수만큼으로 log joint probability를 나눠주어 최종 스코어를 구하게 된다.

## 3. BLEU score
### 3-1. precision and recall
- precision
    - correct words / 예측한 문장의 길이
    - 우리에게 노출된 정보에 대한 정확도
- recall
    - correct words / 기존 문장의 길이
    - 실제로 노출되어야 할 정보 중 얼마나 나왔는지에 대한 것
    - 의도에 부합하는 정보 중 얼마나 잘 사용자에게 노출시켜주었는가
    - 총 소환하려고자 하는 대상 중 얼마나 소환했는가
- F-measure
    - 서로 다른 기준으로 계산된 값을 하나로 도출
    - 조화평균을 이용해 precision과 recall을 계산한다.
    - 조화평균은 평균을 구하려는 값들 중에서 보다 더 작은 값에 가중치를 두어 평균을 구하는 것이라고 이해할 수 있다.
- 하지만, 위의 metric들은 문장에서의 단어 순서를 고려하지 않기 때문에 보다 덜 정확할 수 있다.
- 이러한 issue를 보완하기 위해 주로 기계 번역 task에서 제안된 성능 평가 measure로서 BLEU score를 많이 사용한다.

### 3-2. BLEU score
- BiLingual Evaluation Understudy
    - N-gram overlap between machine translation output and reference sentence
    - Compute precision for n-grams of size one to four
    - Add brevity penalty (for too short translations)
![39](https://user-images.githubusercontent.com/53552847/132700118-60c49e1e-88f9-4fb0-9a6d-5f7f9cc3cc87.PNG)
    - Typically computed over the entire corpus, not on single sentences
- 개별 단어 level에서 봤을 때, 얼마나 공통적으로 groudn truth 문자오가 겹치는 단어가 나왔느냐에 대한 계산뿐만 아니라 N-gram이라 불리는 연속된 N개의 단어에서 봤을 때, 그 문구 혹은 phase가 ground truth와 얼마나 겹치는가를 계싼해서 최종 평가 measure에 반영한다.
- BLEU score는 기본적으로 번역에서 precision만을 고려한다. 즉, recall은 무시하게 되는데 precision의 특성상 번역 문장이 기존의 ground thruth 문장에 비해 빠진 문구가 있다 하더라도 번역 문장만으로도 충분히 의미를 전달할 수 있기 때문이다.
- 더불어 가 n-gram별 precision 값을 기하평균 내어 사용한다.
- 조화평균의 경우, 작은 값에 지나치게 큰 가중치를 주는 특성으로 인하여 기하평균을 사용하는 것으로 이해할 수 있다.
- 위의 식에서 min 부분은 brevity penalty를 의미하는데, 이는 ground truth의 길이를 기준으로 예측 문장이 더 짧아졌을 경우 precision값의 기하평균을 낮춰 사용하기 위해서이다. ground truth에 비해 예측 문장이 짧아질수록 precision은 보다 높은 값을 취하기 때문이다.
![40](https://user-images.githubusercontent.com/53552847/132700121-8ad9f472-a5f6-443a-9822-4b9294fa4f06.PNG)
- 위의 문장에서와 같이, 예측 문장이 ground truth에 비해 더 길 경우는, n-gram으로 인한 penalty를 이미 받고 있기 때문에 이 부분까지 고려할 필요는 없다.

## 4. 피어 세션
- dictionary를 쓰면 안 되는 이유는?
- <EOS>를 쓰는 이유는?
- `decoded.view(-1,self.ntoken)`를 언제 써야 하는가?
- encoder() 부분이 input을 받아서 어떻게 처리하는지 잘 모르겠다.
- 과제 1에 대해서 논의를 했다.

## 5. 27일차 후기
갑작스레 등장한 팀원 모집.. 사실 공부하는 것도 벅찬데 앞으로 함께할 팀원을 모집하라니 이게 무슨 날벼락인가 싶었다 (흑흑)

다른분들은 자기가 어떤 사람이고 어떤 사람을 원한다는 글을 올리는데 명함이나 내밀 수 있을 지 모르겠다. 아무나 날 좀 데려가ㅠㅠ

나서서 적극적인 스타일은 아니다 보니 이런 자율적인 팀원 모집에서 흠이 발생한 것 같다. 갑작스런 일에 살짝 멘탈이 흔들리긴 했지만 일단 맡은 일에 최선을 다하고 기초에 충실해야 겠다. 

어떻게든 되겠지! 일단은 열공하고 좋은 사람 만나기를 기도하자!!

## 6. 해야할 일
- decoder hidden state vector는 output layer의 입력으로 사용되고 동시에 입력 sequence에서 주어지는 각 word 별로의 encoding vector 중 어떤 단어 vector를 중점적으로 가져와야할 지, attention 가중치를 결정해주는 역할을 수행학게되는데 이 때, 다음 time step의 input으로 들어가 총 3가지 역할을 하지 않는가?
- BLEU score가 번역 문장 평가에 있어서 갖는 단점은 무엇인가?
