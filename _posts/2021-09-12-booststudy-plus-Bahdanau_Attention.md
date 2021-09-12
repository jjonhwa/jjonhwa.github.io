---
layout: post
title: "bahdanau attention 논문 리뷰"
categories: booststudy
tags: plus
comments: true
---
Neural Machine Translation By Jointly Learning to Align and Translate 에 대하여 설명한다.

본 내용은 [논문](https://arxiv.org/abs/1409.0473)을 바탕으로 작성했다.

## 목차
- [1. Introduction](#1-introduction)
- [2. Background: Neural Machine Translation](#2-background-neural-machine-translation)
- [3. Learning to align and translate](#3-learning-to-align-and-translate)
- [4. Experiment settings](#4-experiment-settings)
- [5. Results](#5-results)
- [6. Related Work](6-related-work)
- [7. Conclusion](7-conclusion)

## 1. Introduction
- phrase-based translation과 달리, 별도로 tuning되는 작은 sub-components로 구성된 Nueral Machine Translation은 하나의 single, large Neural Network를 Build하고 학습시켜 문작을 읽고 올바르게 번역된 결과를 output한다.
- 이전에 제안된 대부분의 Neural Machine Translation Model은 encoder-decoder family에 속하고, encoder neural network는 source 문장을 고정된 길이의 벡터로 encode하고 decoder는 encoded vector를 번역하여 출력한다. 
- issue
    - encoder-decoder 방법의 접근으로서 Neural Network는 source 문장을 고정된 길이의 벡터로 변환함으로서 모든 정보를 하나의 벡터에 압축할 필요가 있다.
    - 이로 인해, Neural Network가 긴 문장에 대응하는 것을 어렵게 만든다. 실제로, training corpus보다 더 긴 문장을 다룰 경우, 훨씬 예측하기 어렵다.
    - Basic Encoder-Decoder가 input 문장의 길이가 길어짐에 따라 빠르게 변질된다는 것을 Cho et al.(2014b)에서 부터 확인할 수 있다.
- **위의 issue를 해결하기 위해, jointly하게 alignment와 translation을 학습할 수 있는 확장된 형태의 encoder-decoder model을 소개한다.**
    -  각 단어를 생성할 때마다, 가장 관련성이 높은 정보가 집중된 source sentence의 위치 set에 대하여 search한다.
    -  이전의 생성된 모든 target word와 source position과 관련된 context vector를 기반으로 target word를 예측한다.
    -  여기서는 전체 source 문장을 하나의 고정된 길이의 vector로 만들지 않고, sequence of vector의 subset을 만들고 번역이 진행하는 동안 adaptively하게 subset을 선택하고 이를 고려하여 번역을 한다.
-  Align과 Translation을 동시에 학습하는 방법이 기존의 basic encoder-decoder 접근에 비해 훨씬 성능이 좋음을 볼 수 있다.
-  Qualitative Analysis에서 제안된 모델이 source 문장과 대응되는 target 문장 사이에서 언어적으로 타당한 soft alignment를 발견했음을 보인다.

## 2. Background: Neural Machine Translation
- 확률론적 관점에서 translation은 source sentence x가 주어졌을 때, target word의 조건부 확률을 최대로 하는 target sentence y를 찾는 것과 동일하다.
- Neural Machine Translation의 모델 훈련 및 예측 과정은 Parallel training corpus를 사용하여 문장쌍의 조건부 확률을 최대화하는 Parametered Model을 학습시킨다. 즉, Conditional distribution이 translation model에 의해 학습이되고, souorce sentence에 대한 conditional probability를 최대화하는 문장을 찾음으로서 예측을 수행한다.
- 다수의 논문에서 앞서 말한 conditaionl distribution을 직접 학습하기 위하여 Neural Network 사용을 제안했고, 이러한 Neural Machine Translation의 접근은 encoder와 decoder를 사용하는 것이다. 즉, 다양한 길이의 source sentence를 하나의 고정된 길이의 vector로 encoding하고, 이를 다시 다양한 길이의 target sentence로 decoding한다.
- LSTM에 기반한 Neural Machine Translation은 phrase-based machine translation system의 SOTA 성능에 근접했고 이는, phrase pair의 score를 내거나, candidate translation 중 rerank하는 것에 대하여 Neural Components를 더하는 것이 이전의 SOTA 성능을 능가할 수 있게 한다.

### 2-1. RNN Encoder-Decoder
- Align과 Translate을 동시에 학습할 수 있는 새로운 Architecture인 RNN Encoder-Decoder라고 불리는 모델에 대하여 간략하게 기본적인 틀을 설명한다.
- encoder
    - input sentence인 sequence of vector X를 vector c로 변환한다.
    - RNN에서 사용하는 대부분의 접근은 다음과 같다.
![1](https://user-images.githubusercontent.com/53552847/132978135-c8c2c63e-52bd-490b-a40b-dfb6ab3b0f91.PNG)
![2](https://user-images.githubusercontent.com/53552847/132978139-926600ba-a7ef-4412-aee6-e9e4cf5ad512.PNG)
    - $$h_t$$ 는 time t에 대한 hidden state vector를 의미하고, c는 sequence of the hidden state vectors로 부터 생성된 vector이다. f, q는 nonlinear function이다.
- decoder
    - context vector c와 이전 예측 단어 $$y_{t-1}$$를 입력으로 받고, $$y_t$$를 예측하게 된다.
    - 다음의 식과 같이 결합확률을 조건부확률로 분해함으로서 translation y에 대한 확률을 정의한다. RNN에서와 마찬가지로 각 조건부 확률은 다음의 식으로 부터 modeling된다.
    ![3](https://user-images.githubusercontent.com/53552847/132978272-4ef2f13b-0e5c-454a-8d60-f23ba533a606.PNG)
    ![4](https://user-images.githubusercontent.com/53552847/132978273-2c9372cf-c72e-484a-bde4-e241a415fced.PNG)
    - g는 $$y_t$$의 확률을 출력하는 non-linear, potentially multi-layered function이다.
    - $$s_t$$는 Decoder RNN의 hidden state vector이다.
 
## 3. Learning to align and translate
- Neural Machine Translation에서의 새로운 Architecture을 제안한다.
- **새로운 Architecture는 Encoder가 Bidirectional RNN으로 구성되어 있고, Decoder는 번역을 decoding하는 동안 source sentence에서 검색을 진행하는 방식으로 구성된다.**
- 전체적인 모양은 다음 그림과 같다.
![4-1](https://user-images.githubusercontent.com/53552847/132978885-ad187c5e-384c-4dc3-9991-05d73247373a.PNG)

### 3-1. Decoder: General Description
- 새로운 Model Architecture로서, Conditional Probability를 다음과 같이 정의하고, time step i에 대한 Decoder RNN hidden state vector인 $$s_i$$ 역시 다음과 같다.
![5](https://user-images.githubusercontent.com/53552847/132978452-3da145bf-afae-48b4-bd02-8c4381dc3e00.PNG)
![6](https://user-images.githubusercontent.com/53552847/132978453-240bba99-624f-433e-80fa-8b2ca6384299.PNG)
- 여기서 기존의 encoder-decoder에서의 decoder와 다른 점은 각 target word $$y_i$$를 예측할 때의 확률은 별개의 context vector $$c_i$$에 의해서 조건화 된다는 것이다.
- context vector $$c_i$$는 input sentence를 매핑시키는 sequence of annotations $$h_1, ..., h_t$$ 에 의존하는데, 이 때 $$h_i$$를 i번쨰 annotation이라 하고, 각각의 annotation들은 각 input word를 둘러싼 부분에 강한 focus를 가지면서 input sequence 전체에 대한 정보를 포함하고 있다.
- context vector $$c_i$$는 다음의 수식과 같이 $$h_i$$들의 weighted sum으로 계산이 되고, $$h_j$$에 대한 각각의 가중치 $$\alpha_{ij}$$ 는 다음의 수식과 같이 계산된다.
![7](https://user-images.githubusercontent.com/53552847/132978932-ee40ed06-b362-4e45-9ac0-99a52504d297.PNG)
![8](https://user-images.githubusercontent.com/53552847/132978930-95816449-9a86-4c91-bd28-1479d24a5c73.PNG)
- 위의 식에서 $$e_{ij}$$ 는 가중치를 구할 때 사용하는데, position j 주변의 입력과 position i의 출력이 얼마나 일치하는지에 대한 정도를 평가하는 alignment model이며, Score는 Decoder RNN hidden state vector인 $$s_i$$와 input sentence의 $$j_{th}$$ annotation $$h_j$$에 근거한다.
- Alignment Model a를 제안된 system의 모든 다른 구성요소와 함께 jointly trained되는 feedforward neural network로서 매개변수화 시킨다.
- 전통적인 Machine Translation 방법과 달리, 본문에서 제안된 모델은 alignment가 latent variable로서 고려되지 않고 직접적으로 학습시켜서 cost functino의 gradient로서 soft alignment를 계산한다.
- 모든 annotation들의 weighted sum을 취하는 것은 모든 position에 대한 기대되는 position을 계산하는 접근법으로 이해할 수 있다.
- $$\alpha_{ij}$$ 는 source word $$x_j$$가 target word $$y_i$$로 정렬 및 번역될 확률값으로 볼 수 있고, context vector $$c_i$$는 이러한 $$\alpha_{ij}$$ 와 함께 모든 annotation에 대한 기대되는 annotation이라고 볼 수 있다.
- 확률 $$\alpha_{ij}$$ 또는 관련된 energy인 $$e_{ij}$$ 는 decoder의 next hidden state vector $$s_i$$와 target word $$y_i$$를 결정하는데 있어서 이전 decoder hidden state vector인 $$s_{i-1}$$ 에 대한 annotation $$h_j$$의 중요성을 반영한다. 즉, decoder가 source sentence에서 어느 부분에 집중할 지 결정을 하고 이로부터 encoder가 이전에 고정된 길이의 벡터로 모든 정보를 압축시켜야 하는 과정의 부담에서 벗어나게 해준다. 

### 3-2. Encoder: Bidirectional RNN for annotating sequences
- 제안된 scheme에서는 기존에 순차적으로 읽던 RNN과 달리, 각 단어의 annotation이 이전 단어뿐만 아니라 이후 단어에 대한 summarize 역시 하기 위해서 음성인식에서 성공적으로 사용되고 있는 Bidirectional RNN을 사용한다.
- Bidirectional RNN은 forward & backward RNN으로 구성되어 있고, forward RNN은 순차적으로 input sequence를 읽고 sequence of forward hidden state를 계산하며, backward RNN은 reverse로 input sequence를 읽고 sequence of backward hidden state를 발생시킨다.
- 다음과 같이 각 단어 $$x_j$$에 대한 annotation $$h_j$$를 forward hidden state vector와 backward hidden state vector를 concatenating함으로서 얻을 수 있다.
![9](https://user-images.githubusercontent.com/53552847/132981425-9baedff3-306d-4705-9504-673984c1ae5c.PNG)
- 최근의 input에 대하여 더 잘 표현하는 RNN의 경항성으로 인하여, 위의 annotation $$h_j$$는 $$x_j$$에 대한 주변 단어들에게 focus를 맞추어 진행 할 수 있게 된다.


(추가)
