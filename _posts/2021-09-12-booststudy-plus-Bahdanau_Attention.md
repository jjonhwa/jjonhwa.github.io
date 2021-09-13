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

## 4. Experiment settings
- 위에서 제안된 접근방법을 English-to-French translate task를 바탕으로 평가한다. "ACL WMT '14"에서 제공된 bilingual, parallel corpora를 사용한다.
- 성능의 비교를 위해, 최근에 제안된 RNN Encoder-Decoder와 함께 같은 dataset과 training procedures를 사용하여 진행한다.

### 4-1. Dataset
- Europarl(61M words), news commentary(5,5M), UN(421M) 그리고 90M, 272.5M 크기의 두 개의 crawled corpora를 사용하였고 통합하여 850M words를 사용한다.
- "Cho et al"에 있는 절차를 따르고, "Axelrod et al"에 의한 data selection 방법을 사용하여 348M 개의 단어를 가진 combined corpus의 사지으로 줄여서 사용한다.
- 혹여나, 훨씬 더 큰 단일 언어 말뭉치를 사용하여 encoder를 pre-train 가능할지라도, Parallel corpus를 제외한 어떤 하나의 단일 언어로 된 데이터를 사용하지 않는다.
- validation set은 news-test-2012와 news-test-2013을 concatenate하여 사용하였고, training data에 사용되지 않는 "WMT '14"의 3003개의 문장으로으로 구성된 test set을 만들고 이로 모델을 평가한다.
- **tokeniation을 진행한 후, 각 언어에서 가장 빈번하게 등장한 3만개를 최종 word list로 사용하고, 이에 속하지 않는 단어들은 모두 special token이나 [UNK]로 매핑했다. 더불어, lowercasing or stemming과 같은 특별한 전처리 방법은 데이터에 적용하지 않았다.**

### 4-2. Models
- RNN Encoder-Decoder와 RNNsearch로서 제안된 모델, 두 개에 대하여 모델을 학습한다.
- 각각 두 번 훈련시키는데, 30단어에 이르는 문장으로(RNNencdec-30, RNNsearch-30)으로 진행하고, 50단어에 이르는 문장으로(RNNencdec-50, RNNsearch-50) 진행한다.
- **RNNencdec의 encoder와 decoder는 1000개의 hidden units를 각각 가지고, RNNsearch의 encoder는 forward, backward RNN 각각 1000개의 hidden units을 가지고 decoder 역시 1000개의 hidden units을 가진다.**
- 이 두 case 모두, **각 target word의 조건부 확률을 계산하기 위해서 single maxout hidden layer를 가진 multi-layer network를 사용한다.**
- 각 모델을 학습하기 위해 **Adadelta를 가진 minibatch SGD 알고리즘#**을 사용하고, **minibatch의 크기는 80**으로 사용하여 경사하강을 진행한다. 
- 각 모델은 약 5일 가량 학습했다.
- 학습이 완료된 후, 조건부 확률을 대략적으로 최대화하는 번역을 찾기 위해서 **Beam Search를 사용한다.**
- 모델의 architecture의 더 자세한 세부사항과 실험에 사용된 훈련 과정은 Appendices A, B에서 확인할 수 있다.

## 5. Results
### 5-1. Quantitative Result
![10](https://user-images.githubusercontent.com/53552847/133007230-57b37287-2932-4228-b648-42726c7800c5.PNG)
- 위의 표에서, BLEU score로 측정된 번역 performance를 확인할 수 있다.
- 모든 케이스에서, RNNsearch가 기존의 RNNencdec에 비해 월등한 성능을 가지는 것을 명확히 알 수 있다.
- Dataset을 고려할 때, RNNsearch의 기능이 우리가 사용한 parallel corpora 뿐만 아니라 단일 언어 corpus(418M) (Moses)를 사용하여 학습한 기존의 phrase-based translation system과 유사한 성능이 나온다는 것은 유의미한 성과로 볼 수 있다.
- 제안된 접근의 숨겨진 동기 중 하나는 basic encoder-decoder approach에서의 fixed-length context vector의 사용이 긴 문장에 대한 underperform으로 이끌었다고 추측했다.
![12](https://user-images.githubusercontent.com/53552847/133007351-816ff88e-2de4-47c5-93f0-d5d735c0fef5.PNG)
- 위의 그림에서 볼 수 있듯이, RNNencdec의 성능은 문장의 길이가 길어질수록 극적으로 안좋아지는 것을 볼 수 있다.
- 반면에 RNNsearch-30, RNNsearch-50 모두 긴 문장에 대한 좀 더 robust함을 볼 수 있다. 특히, RNNsearch-50의 경우 길이가 50이상인 문장에 대해서도 견고하다는 것을 볼 수 있다.

### 5-2. Qualitative Analysis
#### 5-2-1. Alignment
- 위에서 제안된 접근 방식은 source sentence에서의 단어들과 번역으로 생성된 단어들 사이의 soft alignment를 점검하는 직관적인 방식을 제공한다.
- 가중치 식으로부터 annotation weight $$\alpha_{ij}$$ 를 다음과 같이 시각화할 수 있다.
![11](https://user-images.githubusercontent.com/53552847/133007232-c3d195bb-ddb5-4d28-a1b2-87f6095f3cc4.PNG)
- 위의 그림으로부터, source sentence의 position들이 target word를 생성할 때 어디에 좀 더 많은 중요성을 두었는지에 대하여 시각적으로 볼 수 있다.
- 특히, 영어와 프랑스어의 순서는 대채로 단조롭고, 형용사와 명사의 순서가 다소 다르게 매겨지는데 이를 정확하게 수행해 내고 있음을 볼 수 있다.
- 더불어, [the man]을 [l' homme]로 번역한다고 했을 때, hard alignment의 경우 [the]와 [l']만을 매핑하여 고려하기 때문에, [the]를 [le], [la], [les] or [l']로 번역하는데 있어서는 도움이 되지 않는다. 이에 soft alignment는 [l']을 이후 단어인 [man]도 고려함에 있어서 이러한 문제를 해결할 수 있다.
- 또한, soft-alignment의 장점은 몇몇 단어 혹은 Null을 매핑할 수 있는 역방향 방법을 요구하지 않고 다른 길이의 source phrases와 target phrases를 다루 수 있다는 점이 있다.

#### 5-2-2. Long Sentence
- 5-1의 그림으로부터, 제안된 모델인 RNNsearch가 긴 문장을 학습하는데 있어서 RNNencdec보다 훨씬 좋다는 것을 알 수 있다.
- 이는 RNNsearch가 긴 문장을 고정된 길이의 벡터로 encoding하지 않고 단지 특정 단어를 둘러싼 입력 문장의 부분들을 정확하게 encoding한다는 사실에 의한 것 같다.
- 제시된 정량적 결과와 함께, 이러한 정량적인 관찰은 RNNsearch Architecture가 표준 RNNencdec 모델에 비해 훨씬 더 신뢰할 수 있는 번역 결과를 가진다는 가설을 확증한다.

## 6. Related Work
### 6-1. Learning to align
- output symbol과 input symbol의 aligning에 대한 유사한 접근 법은 필적 합성의 context에서 최근 제안되었다.
- 주어진 문장에 대해 필적을 적용하는 task로서 annotation의 가중치를 계산하기 위하여 Mixture Gaussian kernel을 사용했고, 각 커널의 위치, width 및 mixture coefficient는 alignment model에 의해 예측되었다.
- 보다 구체적으로 위의 alignment는 위치가 단조롭게 증가하도록 위치를 예측하였는데 이는 올바른 번역을 위한 순서의 재배치를 심각하게 제한한다.
- 이러한 접근법과 달리, 본 논문에 제안한 align 학습 방법은 각 단어의 번역에 있어서 모든 단어의 annotation weight를 계산하여 사용한다. 

### 6-2. Neural Networks for Machine Translation
- 기존의 Neural Network를 이용한 Machine Translation 분야에서는 하나의 feature를 기존의 통계적 기계 번역 시스템에 전달하거나, 번역 후보군들의 순위를 다시 매김하는 것을 크게 제한하였다.
- 이에 대하여, 기존에 존재하는 번역 시스템의 하위 요소로서 Neural Network를 성공적으로 사용하여, 전통적으로 target-side language model로서 학습된 Neural Network는 번역 후보군들을 다시 랭크 매기거나 다시 점수를 매기게 한다.
- 이러한 시도를 바탕으로 Machine Translation System의 SOTA보다 더 좋은 번역 성능을 보였음에도 불구하고 **우리는 Neural Network만을 기반으로한 완전히 새로운 번역 시스템을 만들고자한다. 본 논문에서 고려한 Neural Machine Translation System의 접근은 이러한 초기 번역 시스템으로부터 근본적으로 벗어난 것이다.**
- 즉, 기존 시스템의 일부로서 Neural Network를 사용하는 것 대신, 모델 그자체로서 source sentence로부터 직접적인 번역 결과를 생성한다.

## 7. Conclusion
- encoder-decoder approach라고 불리는 Neural Machine Translation에서의 관례적인 접근은 전체 input sequence를 하나의 고정된 길이의 벡터로 encode하고 이를 바탕으로 decode를 진행하며 번역 결과를 수행하는데, 최근 연구에 의거하여, 이러한 고정된 길이의 context vector를 사용하는 것은 긴 문장을 번역하는데 문제가 있을 것이라고 추측하였다.
- 위의 issue를 다루기 위하여, 새로운 architecture를 제안하는데, 이는 각 target word를 생성할 때, input word set과 encoder에 의해 계산된 annotation을 모델이 search할 수 있도록 basic encoder-decoder를 확장했다.
- 이로부터, encoder가 전체 source 문장의 정보를 하나의 고정된 길이의 벡터로 압축하는 것을 하지 않게 만들 수 있었고, 모델이 target word의 생성에 관련있는 정보에만 집중할 수 있게 만들 수 있다.
- 전통적인 Machine Translation System과 달리, alignment mechanism을 포함한 번역 시스템의 모든 pieces들은 정확한 번역을 생산하기 위해 더 높은 log probability를 만드는 방향으로 jointly하게 학습된다.
- English-to-French task에서 제안된 모델인 RNNsearch는 문장의 길이와 상관없이 RNNencdec에 비해 상당히 높은 성능을 가졌고, source sentence의 길이에 대해 훨씬 더 robust함을 실험적으로 볼 수 있었다.
- alignment에 대한 정량적인 분석에 의해서, 모델이 각 target 단어를 관련 source word 또는 그들의 annotation과 함께 정확하게 정렬할 수 있다는 결론을 낼 수 있었다.
- 좀 더 중요하게, 제안된 접근은 기존의 phrase-based machine translation과 비교할만한 번역 성능을 달성했고, 이는 Architecture가 전부 Neural Machine Translation으로 이루어져있다는 것을 고려할 때 엄청난 결과이다.
- 앞으로 도전할 것들 중의 하나는 rare word 혹은 unkown words를 더 잘 다루는 것이다. 이는 모델이 더 널리 사용되거나 모든 context에서 기존의 SOTA Machine Translation System의 성능을 대신하기 위해 요구될 것이다.

본 리뷰는, 'Neural Machine Translation By Jointly Learning to Align and Translate'을 바탕으로 번역 및 본인 스스로의 의견을 덧붙여 작성하였으며 잘못된 내용이 있다면 의견을 남겨주시면 감사하겠습니다!
