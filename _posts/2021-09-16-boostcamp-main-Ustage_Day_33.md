---
layout: post
title: "NLP Ustage Day 9 (Day 33)"
categories: boostcamp
tags: main
comments: true
---
Self-supervised Pre-training Model에 대하여 학습한다.

**부스트 캠프 33일차 학습 요약**
- **학습** : Self-supervised pre-training models
- **행사** : 오피스아워
- **피어 세션**

## 목차
- [1. Self-supervised pre-training models](#1-self---supervised-pre---training-models)
- [2. GPT-1](#2-gpt---1)
- [3. BERT](#3-bert)
- [4. BERT vs GPT](#4-bert-vs-gpt)
- [5. Task(benefit with BERT)](#5-taskbenefit-with-bert)
- [6. 33일차 후기](#6-33일차-후기)
- [7. 해야할 일](#7-해야할-일)

## 1. Self-supervised pre-training models
- GPT-1과 BERT를 소개한다.
- GPT-1과 BERT는 Tranfer Learning, Self-supervised Learning, Transformer를 사용했다는 공통점이 있다.
- 위의 세 가지의 강력한 무기를 이용해 대용량의 text를 학습한 모델을 target task에 적용해 거의 모든 기존 자연어처리 task를 압도하는 성능을 가졌다.

### 1-1. Recent Trends
- Transformer 및 self-attention block은 어떤 범용적인 sequence encoder, decoder로서 최근 NLP의 다양한 분야에서 좋은 성능을 내고 있으며, 심지어 다른 분야까지 활발하게 사용되고 있다.
- Transformer에서는 self-attention block을 6개 정도만 쌓아서 사용햇다면, 최근 모델의 발전 동향은 Transformer에서 제시된 self-attention block을 점점 더 많이 쌓아서 모델 구조 자체의 특별한 변경 없이 self-attention block을 deep하게 쌓은 모델을 만들고 있다.
- 이렇게 self-attention block을 점점 더 많이 쌓는 형태의 대규모 모델과 함께 대규모 학습 데이터를 통해 학습할 수 있는 self-supervised learning framework로서 학습한 후에 이를 다양한 task들의 transfer learning의 형태로 fine-tuning 하는 형태로 이용해 좋은 성능을 내고 있다.
- self-attention model은 추천시스템, 신약개발, 영상처리 분야에 까지도 확장하고 있다.
- Self-attention에 기반한 Transformer와 같은 모델들도 자연어 생성 task에서 단어를 처음부터 왼쪽에서 하나씩 생성한다는 greedy decoding이라는 framework에서 벗어나지 못하고 있고, 이를 기반해서 생긴 모델이 GPT이고, 이를 벗어나 Bidirectional의 형태를 사용하여 만든 모델이 BERT이다.

## 2. GPT-1
- 일론머스크의 비영리 연구기관인 OpenAI에서 나온 Model이다.
- 자연어 생성에서 놀라운 결과를 보여주었다.
![24](https://user-images.githubusercontent.com/53552847/133711807-77a5fe82-fda5-4db0-be54-e358ae8665f4.PNG)
- 위의 그림의 가운데에서 볼 수 있듯이, 기존의 token화된 문장에 Special token을 추가하는 것을 제안함으로서 simple한 task 뿐만 아니라 다양한 자연어 처리에서의 많은 task들을 동시에 커버할 수 있는 통합된 모델을 제안했다.

### 2-1. 모델 구조
- Text Sequence에 Position Embedding을 더한 후, Self-attention block을 총 12개 쌓는다.
- 위의 그림의 왼쪽에서 볼 수 있듯이, Text Predicion task 즉, Language Modeling task를 통해 전체 12개의 Self-attention block으로 이루어진 GPT-1 Model이 학습된다.
- GPT-1은 Language Modeling task 뿐만 아니라 문장 level 혹은 다수의 문장이 존재하는 경우에 대한 task에 대해서도 모델의 큰 변형없이 활용될 수 있도록 하는 학습 framework를 제시한다.
- Classification task를 예로 들어, 위쪽의 그림에서 볼 수 있듯이, 이 경우 문장의 맨 앞에 start token을 넣고 마지막에 extract token을 넣어서 문장의 encoding vector를 만드는데, 이 때 extract token에 해당하는 encoding vector를 최종 output layer의 input vector로 줌으로서 classification으로 분류하는 task로 학습한다.
- 즉, 주어진 문장이 있다고 할 때, Transformer 모델을 통해 각 word 별로 encoding을 진행하고 extract token에 대한 vector만을 Linear Transformer을 통해 classifciation하는 task로서, 다음 단어를 예측하는 LM과 동시에 classification을 위한 labeled된 데이터를 동시에 학습할 수 있다.
- 마찬가지로 entailment task를 예로 들면, entailment task는 premise, hypothesis에 해당하는 문장의 논리적인 내포관계를 찾는 task(premise가 참일 경우 hypothesis가 참 혹은 모순 등)인데, 이 task의 경우 다수의 문장으로 이루어진 입력을 받게되는데, 두 문장을 문장 사이에 delimiter를 삽입하고 classification과 같이 문장 처음에는 start token을 마지막에는 extract token을 삽입한 채로 input vector를 만들고 이를 encoding vector를 만든다.
- classification과 같이 extract token을 최종 output layer에 통과시켜주고 이를 바탕으로 premise, hypothesis가 실제로 논리적인 내포 혹은 모순관계인지에 대하여 분류하는 task를 수행한다.

### 2-2. 학습 방법
- extract token의 Query가 self-attention을 통한 여러 정보들을 주어진 입력 문장들로 부터 적절하게 추출하는 역할을 한다.
- GPT-1을 다른 task에 transfer learning의 형태로 활용할 때는, 기존의 학습된 task와는 다른 결과를 output으로 가지기 때문에 기존의 예측 task를 진행하는 output layer를 떼어내고 그 전단계에서 출력된 encoding vector를 바탕으로 새로운 task를 위한 추가적인 output layer를 붙이고 이를 random initialization을 하여 main task를 위한 학습데이터로 전체 Network를 학습한다.
- 마지막 layer의 경우 random initialization이 되어 충분히 학습되어야 하지만 이전에 있던 학습된 layer들은 learning rate를 상대적으로 작게 주어 큰 변화가 일어나지 않도록 학습한다.
- Language Modeling은 별도의 label을 필요로 하지 않는 task이므로 굉장히 많은 양의 데이터를 통해서 이러한 task를 통해 모델을 학습시키고, main task에서는 label이 부여되어 있어야 하므로 상대적으로 소량인 main task에서의 data를 바탕으로 fine-tuning하여 모델을 task에 맞게 학습한다.
- 이렇게 별도의 label을 필요로 하지 않는 pre-training을 거치는 모델이기 때문에 Self-supervised pre-training model이라고 부른다.

## 3. BERT
![25](https://user-images.githubusercontent.com/53552847/133711810-bb67cc10-43cd-45de-9bb0-d442d7046b3f.PNG)
- 현재까지도 가장 널리쓰이는 Pre-Training 모델이다.
- GPT와 마찬가지로 LM task를 바탕으로 pre-training 모델을 수행한다.
- BERT의 핵심 기술은 MLM(Masked Language Model)을 통한 bidirectional 형태의 encoding을 사용하는 방식과 pre-training task를 추가한다는 것이다.

### 3-1. MLM(Masked Language Model)
![26](https://user-images.githubusercontent.com/53552847/133711811-08abd03b-75e9-4935-b851-0ba581985deb.PNG)
- Motivation : Language models only use left context or right context, but language understand bi-directional
- 즉, 기존의 모델들은 앞서 등장한 단어를 바탕으로 이후에 등장할 단어를 예측하는 방식으로 pre-training을 하였는데, 단어를 예측할 때에는 앞 뒤 문장의 문맥을 바탕으로 예측해야한다.
- 어떤 문장이 주어졌을 때, 각 단어에 대하여 일정한 확률로 [MASK]라는 token으로 치환하고 이 [MASK]라는 token이 어떤 단어인지를 맞추는 방식으로 모델이 학습된다.
- 몇 %의 단어를 [MASK]로 치환할 것인지는 Hyper-parameter 이지만 BERT 논문상의 결과로부터 흔히 15%를 [MASK]로 치환하여 진행한다. 이 때 [MASK]의 비율을 너무 많이 가지게 되면 정보의 손실로 인한 문제가 발생하고, 너무 적게 가지게 되면 전체적인 학습 효율이 떨어지고 학습속도가 느려질 수 있다.
- 15%의 단어들을 [MASK]로 전부 변환했을 때의 문제점
    - pre-training 할 때에는 모델이 [MASK]에 대하여 예측함으로서 [MASK]를 포함한 문장에 익숙해지게 되는데, downstream task에 이 모델을 적용할 경우 [MASK]라는 token은 등장하지 않게 되므로, train/test 에서 상당히 다른 양상을 띄거나 다른 pattern을 보일 수 있다. 이로서 transfer learing의 효과를 끌어올리는 데에 문제를 야기할 수 있다.
    - 이러한 문제에 대한 대안으로서, 예측할 15%의 단어 전부를 [MASK]로 치환하지 않고 다음의 방식으로 치환하게 된다.
        - 80%는 [MASK]로 변경하여 예측 -> 원래의 단어를 추출하도록 학습
        - 10%는 다른 단어로 변경하여 예측 -> 원래의 단어를 추출하도록 학습
        - 10%는 원래 단어를 그대로 입력하여 예측 -> 원래의 단어를 추출하도록 학습

### 3-2. pre-training task: Next Sentence Prediction
![27](https://user-images.githubusercontent.com/53552847/133711812-d0a45763-f169-4388-bf1c-fbf95f02e41e.PNG)
- BERT에서는 pre-training 기법으로서 MLM 이외의 문장 level의 task에 대응하기 위한 pre-training 기법도 제안하였다.
- 문장 간의 관계를 학습하기 위하여, 두 개의 문장 A, B에 대하여 B가 A다음에 오는 문장인지 혹은 random한 문장인지를 예측한다.
- 이 task의 경우, 주어진 하나의 글에서 위의 이미지에서 처럼 두 개의 문장을 뽑고 이 두 문장을 [SEP] token을 추가하여 하나의 문장처럼 이어주고 문장의 마지막에도 [SEP] token을 추가한다. 동시에, 문장 level에서의 예측 task를 수행하기 위해 문장의 맨 처음의 [CLS] token을 추가하여 이를 활용하여 output layer에서 classification을 진행한다. 이는 앞서 설명한 GPT에서의 extract token과 같은 역을 하는데 앞쪽에 있다는 차이만 존재한다.
- 별도의 문장 level에서 label이 필요없는 task로서 pre-training하기 위하여 연속적으로 주어진 두 개의 문장이 연속적으로 나오는 문장인지 아닌지를 예측하는 Next Sentence 인지 아닌지에 대한 binary-classification task를 추가해주었다,
- [CLS] token에 해당하는 encoding vector가 output layer를 거치게하여, binary-classification으로서 해당 분류의 ground truth를 실제로 그 두 문장이 인접했는지 안했는지를 예측하는 task로 만들었다.

### 3-3. BERT SUMMARY
![28](https://user-images.githubusercontent.com/53552847/133711813-879bc2ff-3583-49e8-a8b8-9de51ed34c6e.PNG)
- 모델 구조 자체는 Transformer에서 제안된 self-attention block을 그대로 사용했다.
- 위의 이미지에서 Model Architecture 부분의 L, A, H는 각각 self-attention block의 개수, 각 layer별 multi-head의 개수, encoding vector dimension(hidden state dimension)을 의미한다.
- Input sequence를 넣어줄 때, word 단위로 embedding vector를 사용하는 것이 아니라 word를 좀 더 잘게 쪼갠 subword 단위별로 embedding하고 이를 input vector로서 넣어준다.
- positional embedding을 Transformer에서 사용한 사전에 정의된 vector를 사용하는 것이 아닌 word2vec에서 embedding matrix를 학습한 것처럼, random initialization에 의해서 전체 모델 학습 과정에서 end-to-end로 각 position에 더해주어야 하는 positional embedding vector를 학습에 의해 최적화된 값으로 도출할 수 있도록 한다.
- 앞서 설명했던 [CLS] token과 [SEP] token이 추가적으로 삽입된다.
![29](https://user-images.githubusercontent.com/53552847/133711814-772a43d8-caef-483c-bfb4-6424d6b1777c.PNG)
- 이 뿐만 아니라 위의 그림처럼, Segment Embedding이 추가되었는데, 이는 여러 문장이 합쳐져서 하나의 input이 만들어질 경우, 각 문장을 구분할 수 있도록 하는 역할을 한다.

### 3-4. Fine-tuning process
![30](https://user-images.githubusercontent.com/53552847/133711816-98c63895-5a6b-4282-b358-86496de45f49.PNG)
#### Sentence Pair Classification
- 두 개의 문장을 [SEP] token을 기준으로 하나의 문장으로 만든다.
- BERT 모델을 통해 인코딩을 진행한다.
- 각각의 word에 대한 encoding vector를 얻고, [CLS] token에 해당하는 encoding vector를 output layer의 입력으로 주어서 다수 문장에 대한 예측 task를 수행할 수 있다.

#### Single Sentence Classification
- 한 번에 하나의 문장을 입력으로 준다.
- 문장을 하나만 주고 맨 처음에 [CLS] token을 추가하여 이 토큰의 encoding vector를 output layer의 입력으로 주어 classification을 수행한다. 즉, 입력을 하나만 주는 task로 가장 간단하고 그 외의 방식은 sentence pair classification과 동일하다.

#### Single Sentence Tagging Task
- [CLS] token을 포함하여 각각의 단어에 대한 encoding vector가 얻어지면 각각의 vector를 동일한 output layer를 통과시켜서 각 word별 classification 혹은 prediction을 수행한다.
- 각 word의 역할을 추출해내야하는 task로서 모든 단어에 대한 encoding vector가 output layer의 input으로 들어가게 된다.

#### Question Answering
- [5. Task(benefit with BERT)](https://jjonhwa.github.io/boostcamp/2021/09/16/boostcamp-main-Ustage_Day_33/#5-taskbenefit-with-bert)에서 추가적으로 설명하도록 한다.

## 4. BERT vs GPT
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>

||BERT|GPT-1|  
|---|---|---|  
|pre-training task|MLM|LM|  
|Training-data size|BookCorpus and Wikipedia(2,500M words)|BookCorpus(800M words)|  
|Training Special Tokens|[SEP], [CLS], segment embedding|[Extract], delimiter|  
|Batch Size|128,000 words|32,000 words|  
|Learning Rate|task specific fine tuning learning rate|5e-5 for all fine tuning experiments|  
{: .tablelines}

- 일반적으로 더 큰 batch_size를 사용하게 되면, 최종 모델 성능이 더 좋아지고 학습도 안정화 된다. 이는 Gradient Descent를 진행할 때, 일부의 데이터만으로 도출된 gradient로 직접적인 parameter를 업데이트 할 지 혹은 더 많은 데이터를 바탕으로 도출된 gradient로 parameter를 업데이트 할 지에 대한 것으로, 일반적으로 더 많은 데이터를 사용하여 최적화를 수행할 때, 학습이 더 안정적이고 성능이 좋다.
- 하지만, batch size가 커질수록, 메모리가 비례해서 증가하기 때문에 더 많은 GPU 메모리와 고성능의 GPU가 필요하게 된다.
- 기존에 pre-training을 하기위한 output layer를 제거하고, main task를 위한 output layer를 추가하여 학습을 진행하게 되는데, 이 때 output layer는 random initialization을 통해 main task를 위한 데이터를 바탕으로 학습하고, 기존에 학습된 transformer encoder는 상대적으로 적은 learning rate를 사용하여 조금만 변화하도록 하여 일반화 가능한 지식이 최대한 많이 유지될 수 있는 형태로 학습이 진행된다.

## 5. Task(benefit with BERT)
### 5-1. MRC(Quesntion Answering)
- BERT를 통해 더 높은 성능을 fine-tuning을 통해 얻을 수 있는 대표적인 task로서 MRC 즉 machine reading comprehension task가 있다.
- 기본적인 질의 응답 형태이지만, 다음의 그림처럼, 어떤 질문이 주어지고 그 질문에 대한 답을 내는 것이 아니라 기계 독해 즉 주어진 지문을 이해하고 질문을 이해하여 이를 바탕으로 지문으로부터 질문에 대한 정답을 예측해내는 task이다.
![31](https://user-images.githubusercontent.com/53552847/133711818-958c8576-0cd5-4905-a609-1f84c7b407c3.PNG)
- 위와 같이, 주어진 document로부터 질문인 Daniel의 위치를 찾아서 정답을 나타내는 task이다.

### 5-2. SQuAD
![32](https://user-images.githubusercontent.com/53552847/133711820-4d7999e3-df8a-47af-9f03-ed10f5420077.PNG)
![33](https://user-images.githubusercontent.com/53552847/133711822-b8a0e2e0-c0b0-4e59-83e0-25775fe8079d.PNG)
- SQuAD는 MRC를 수행하기 위한 실제로 더 어렵고 유의미한 수준의 Dataset을 의미한다.
- 위의 그림에서 보이는 데이터는 실제로 crowdsourcing을 통해 많은 사람들로부터 task를 수행하도록 하여 수집되어 구축된 데이터이다.
- 주어진 지문과 질문을 서로 다른 문장인 것처럼 [SEP] token을 통해 concat되어 하나의 sequence로 만들어 encoding을 진행한다.
- 이 후, 각각의 지문 상에서 word별로 word encoding vector들이 출력되고, 이들로부터 정답에 해당할 법한 위치를 예측하도록 한다.
- 각 word별로의 encoding vector를 공통된 output vector를 통해 scalar를 도출하도록 하여 이로부터 지문 속에서 답에 해당하는 문구가 시작하는 위치를 예측하도록 한다.
- output layer는 단순히 특정 dimension으로 이루어진 vector를 단일한 차원 혹은 scalar로 나오게 해주는 fully connected layer가 된다.
- scalar를 각 word별로 출력해준 후, 여러 단어들 중에서 답에 해당하는 문구가 어느 단어에서 시작하는 지를 예측하고, 해당하는 scalar값을 softmax를 취하여 softmax의 ground truth로서 시작하는 단어에 배정될 확률이 100%가 되도록 softmax loss를 통해 모델을 학습한다.
- 더불어, 정답 단어의 마지막 위치도 예측해야하는데, starting position을 예측하도록 하는 output layer로서 fully connected layer를 통과함을 동시에 두 번째 fully connected layer를 통과시켜 앞서 말한 방법과 같이 이번에는 끝나는 위치에 해당하는 word를 ground truth로 두어 softamx loss를 통해 모델을 학습시킨다.
- 더 나아가, SQuAD 2.0에서는 항상 정답이 있을 법한 질문을 뽑고, 이 질문에 대한 정답이 없는 Dataset까지도 원래의 Dataset에 포함시킨 Dataset이다.
- SQuAD 2.0은 먼저 질문에 대한 답이 있는지 없는지를 예측하고 답이 있다면 앞서 설명한 방식으로 최종 문구를 예측하게 된다. 이렇기 때문에, 실제로 주어진 질문과 지문에 대하여 정답이 있는지 없는지를 판단하는 task와 정답이 있을 경우 어떤 문구가 정답이 될 지를 예측하는 task를 수행하게 된다.
- 후자의 경우 앞서 설명한 방식대로 진행되며, 전자는 binary-classification 형태의 task로서 [CLS] token을 질문과 지문을 concat한 sequence에 더하여, 이를 바탕으로 정답이 있는지 없는지 output layer를 거쳐 예측하고, crossentorpy  loss를 통해 학습된다.

### 5-3. BERT: On SWAG
![34](https://user-images.githubusercontent.com/53552847/133711826-0fc0ccd9-f8c9-4687-bc65-87be858ab411.PNG)
- 주어진 문장이 있을 때, 다음에 나타날 법한 적절한 문장을 고르는 task이다.
- 기본적으로 [CLS] token을 사용하고, 주어진 문장과 각 문장들을 concat하여 BERT를 통해 encoding하고, [CLS] token에서 나오는 encoding vector를 Fully Connected layer를 통과시켜 scalar를 도출한다.
- 이렇게 출력된 scalar를 concat하여 softmax의 입력으로 주어 ground truth가 되도록 softmax loss를 통해 전체 모델을 학습하게 된다.

### 5-4. Ablation study
![35](https://user-images.githubusercontent.com/53552847/133711828-dafa7338-f944-4a61-af3b-234b8746a6c6.PNG)
- BERT에서 제안한 모델 사이즈가 있을 때, 그 모델을 점점 더 깊게 쌓고 layer별 parameter를 더 늘리는 방식으로 학습을 진행하면 모델 사이즈를 키울수록 여러 downstream task에 대한 성능도 계속적으로 끊임없이 좋아졌따.
- GPU resource가 허락하는 한 모델 사이즈를 키울 수 있을 만큼 키웠을 때, 점점 더 개선되는 결과를 보여주었다.
- 가능한 모델 사이즈를 더 키울 수 있다면, 이러한 pre-training을 통한 다양한 downstream task에 대한 성능은 점점 더 개선될 것이라고 전망한다.

## 6. 33일차 후기
벌써 뭔가 슬럼프 비스무리한 게 온 것 같다. 좀 더 유하게 공부할 필요성이 있는 것 같고 쫓기든 하지말자~

잘하는 사람들에 속해있으니까 한 없이 초라해짐을 느끼면서 공부를 하고 있지만 위기를 기회로서 잘 이용할 수 있도록 노력해야겠다!


## 7. 해야할 일
- extract token이 어떻게 실제 task를 결정할 수 있는가?
- GPT에서도 LM 이외의 pre-training에서의 다른 task를 사용하여 학습하였는가?
- BERT와 GPT의 learning rate 차이.
- SWAG에서 주어진 문장과 각 문장을 concat하여 scalar를 도출할 때, 각 문장들로부터 나온 scalar를 concat해서 사용하는가?
- BERT의 Masked Language Model의 단점은 무엇인가? 사람이 실제로 언어를 배우는 방식과의 차이를 생각해보며 떠올려보자.
