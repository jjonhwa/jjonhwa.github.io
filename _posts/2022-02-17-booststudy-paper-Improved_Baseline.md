---
layout: post
title: "[논문리뷰] An Improved Baseline for Sentence-level Relation Extraction"
categories: booststudy
tags: paper
comments: true
---
[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373)를 읽고 이에 대하여 논의한다.

## 목차
- [1. Abstract](#1-abstract)
- [2. Introduction](#2-introduction)
- [3. Method](#3-method)
- [4. Experiments](#4-experiments)
- [5. Conclusion](#5-conclusion)

## 1. Abstract
- Sentence-level Relation Extraction은 한 문장에서 두 entity 사이의 관계를 식별하는데 초점을 맞춘다.
- 기존의 Relation Classification의 성능에 영향을 미쳤던 "ENTITY REPRESENTATION"과 "NOISY OR ILL-DEFINED LABELS"에 대하여 다시 살펴본다.
- Improved RE Baseline Model은 **Entity Representation을 typed marker와 통합**한다.
- TACRED에서 F1 score 74.6을 달성하며, 이는 기존의 SOTA의 성능보다 훨씬 높은 수치이다.
- 편향되어있는 TACRED와 TACREV를 개선한 Re-TACRED datset에서 91.1%의 F1 Score를 달성한다.

## 2. Introduction
- Relation Extraction에서는 사전에 정의된 관계로부터 주어진 text 내에서 두 entity들의 관계를 식별하는 task이다.
    - 예를 들어, "Bill Gates founded Microsoft together with his friend Paul Allen in 1975"라는 문장과 entity pair ("Bill Gates", "Microsoft")가 주어질 때, "ORG:FOUNDED_BY"라는 관계를 예측해야한다.
- Sentence-level Relation Extraction task는 2가지 연구로 나뉘어진다.
    - Pre-Trained Language Model에 External Knowledge를 주입하는 방식
        - ERNIE와 KnowBERT를 포함한 방법론에서는 Knowledge Graph에서 Pre-Trained된 Entity Embedding을 Transformer의 입력으로 넣는다.
        - 유사한 방식으로, K-Adapter에서도 factural과 linguistic knowledge를 langauge model로 통합하는 plug-in neural adaptor를 소개한다.
        - 더불어, LUKE는 masked language modeling의 pre-training objective를 entity로 확장하며, entity-aware self-attention mechanism을 제안한다.
    - Relation-Oriented Objectives를 사용하여 entity들과 연결된 text를 Pre-Trained Language Model에 fine-tuning하는 것이다.
        - 특히, BERT-MTB는 두 관계 instance가 동일한 entity를 공유하는지 여부를 결정하는 matching-the-blanks objective를 제안한다.
- **Pre-Trained Language Model을 적용한 후에도, TACRED benchmark에서 기존의 SOTA인 70.1%에서 72.7%로 밖에 상승하지 못하였다.**
- Relation Classifier의 성능을 방해하는 두 가지 장애물에 대하여 논의한다.
    - RE task는 raw text와 entity들의 side information으로 구성된 input text를 제공한다.
        - 이때, **Entity들에 대한 side information이 성능에 중요한 것으로 보여진다.**
        - 하지만, 기존에 존재하는 방법들은 text에서 entity에 대한 정보를 포괄적으로 나타내지 못하여 entity의 characterization을 제한하게 된다.
    - TACRED에서 파생된 dataset을 포함하여, **Human-labeled RE datasets은 noise가 있거나 잘못 labeling된 많은 부분들이 있어, 모델 성능의 잘못된 추정의 원인**이 된다.
- Alt et al(2020)에서는 TACRED의 test set을 relabel하였으며 label 중 6.62%가 부정확하다는 것을 발견했다.
- Stoica et al(2021)은 TACRED에서 사전에 정의된 relation의 정의에 대하여 조사하였으며, 그들 중 일부가 모호하다는 것을 발견했고, high-quality label을 보장하기 위하여 improved annotation을 TACRED dataset에 적용하였다.
- 이를 토대로, **Improved RE Baseline을 제안하고, sentence-level RE에서 유망한 typed entity marker를 소개**한다.
- TACRED의 3가지 다른 버전인 Original TACRED, TACREV, Re-TACRED에서 모델을 평가한다.
- Backbone model로서 RoBERTa를 사용하며, improved baseline model은 TACRED에서 F1 score 74.6%, TACREV에서 83.2%, Re-TACRED에서 91.1%를 달성한다.
- 더불어, **TACRED dataset에서의 높은 noise에 대한 비율이 Relation Classifier Model이 entity들에 대한 side information에 의존하도록 한다는 것을 관찰하였고, 이로 인해 biased evaluation의 결과로 이끌어진다.**
- 이러한 문제로 인해서, sentence-level RE task에서 Re-TACRED를 평가 benchmark로 사용할 것을 제안한다.

## 3. Method
- 먼저 Relation Extraction task를 정의한다.
- 그 후, Model Architecture와 Entity Representation Technique을 소개한다.

### 3-1. Problem Definition
- Sentence-level RE에 초점을 맞춘다.
- 구체적으로, Entity pair (es, eo)가 존재하는 문장 x가 주어졌을 때, subject entity와 object entity가 있으며, es와 eo 사이의 관계 r을 예측한다. 
- 관계 r은 R과 NA의 합집합으로 구성된다.
    - 이 떄, R은 사전에 정의된 Relation들의 집합이다.
    - 만약 es와 eo 사이의 관계를 R에 있는 relation으로 표현할 수 없을 경우, NA label을 따른다.
- 예를 들어, entity를 포함하는 문장 "Bill Gates founded Microsoft together with his friend Paul Allen in 1975"과 entity pair ("Bill Gates", "Microsoft")가 주어질 때, "ORG:FOUNDED_BY"라는 관계를 예측해야하며, 만약 "ORG:FOUNDED_BY"가 R에 사전 정의가 되어있지 않을 경우 NA label을 따른다.

### 3-2. Model Architecture
- Pre-Trained Language Model을 사용한 후 fine-tuning한다.
- **Input text "x"가 주어지고, 우선 NER types technique을 활용하여, entity span에 대한 mark를 하며, 이렇게 preprocess된 text를 Pre-Trained Langauge Model에 넣어 학습**한다.
- output으로서 subtoken에 대한 hidden state vector를 취한다.
![1](https://user-images.githubusercontent.com/53552847/154454083-ac5c6d51-a2ec-491c-ad4a-6ad7bbfb3313.png)
    - 이 때, subtoken은 subject entity span과 object entity span에 대한 token을 의미한다.
    - 더불어, **subject entity span과 object entity span에서 subtoken에서 첫번째 subtoken을 최종 embedding $h_{subj}$, $h_{obj}$로 취한다.**
- 이렇게 출력된 최종 embedding을 softmax classifier에 넣어서 예측을 수행한다.
    - 이 때, **$h_{subj}$와 $h_{obj}$를 concatenate하여 FFN을 통과한 후 ReLU를 통과**한다.
    - 그 후, **다시 한 번 FFN을 통과한 후에 softmax를 통과하여 예측 결과값이 출력**된다.
- CrossEntropyLoss를 활용한다.

### 3-3. Entity Representation
- sentence-level RE를 위하여, names, spans, subject entity와 object entity의 NER types를 구조화된 input으로 제공한다.
- 이 때, **entity information은 relation type에 대한 유용한 단서를 제공**한다.
- 예를 들어, **"ORG:FOUNDED_BY"는 subject와 object 유형이 각각 "Organization"과 "Person"일 경우에 유지될 가능성이 높기 때문에, 이를 유용한 단서로 활용할 수 있다.**
- 본 논문에서 기존의 entity representation technique들을 재평가하고 더 나은 방안을 발견한다.
- 모든 새로운 special token에 대한 token embedding은 random하게 초기화되며, fine-tuning 시에 최적화된다.
- type entity marker은 entity names와 entity span, NER type을 표현한 반면 entity marker와 entity mask는 entity type과 entity name을 무시한다.
![2](https://user-images.githubusercontent.com/53552847/154454091-7ad35983-56e2-4959-ba00-cd47a62e797b.png)


#### Entity mask
- 새로운 Special Token인 "[SUBJ-TYPE]"과 "[OBJ-TYPE]"을 활용하여 기존 text에서의 subject, object entity들을 mask한다.
- 이 때, TYPE은 각각의 entity type을 의미한다.
- 이 technique은 PA-LSTM model에서 제안되었으며 후에 SpanBERT와 같은 Transformer based model에서 적용되었다.
- Zhang et al(2017)에서 본 technique이 Relation Classifier를 특정한 entity name에 over-fitting되는 것을 막아주고 좀 더 일반화되는 추론으로 이끌어준다고 주장한다.

#### Entity marker
- 이 technique은 spcial token pair인 [E1], [/E1], [E2], [/E2]를 subject와 object entity에 붙인다.
- 이렇게 해서 input text에 대한 format을 "[E1] SUBJ [/E1] ... [E2] OBJ [/E2]"로 만든다.

#### Entity marker (punct)
- 이 technique은 punctuation을 활용하여 entity span에 앞서 설명한 entity marker와 같이 붙이는 방식이다.
- input text의 경우 "@ SUBJ @ ... # OBJ #"과 같은 형태를 띈다.
- Entity marker와의 주요한 차이는 model이 받는 vocabulary에 새로운 special token을 추가하는지 안하는지 이다.

#### Typed entity marker
- 이 technique은 NER type을 entity marker에 통합하는 방법이다.
- 이는 새로운 Special token인 [S:TYPE], [/S:TYPE], [O:TYPE], [/O:TYPE]을 사용하는 방법이다.
- input text는 "[S:TYPE] SUBJ [/S:TYPE] ... [O:TYPE] OBJ [/O:TYPE]"의 형태가 된다.

### Typed entity marker (punct)
- 새로운 special token을 사용하지 않고, entity span과 entity type을 marking하는 typed entity marker 방식을 제안한다.
- 이는 typed entity marker와 동일하지만 새로운 special token을 사용하는 것이 아니라 @와 #을 사용하며 NER type에 대해서는 *과 ^를 사용하여 추가한다.
- 수정된 text는 "@ * subj-type * SUBJ @ ... # ^ obj-type ^ OBJ #"의 형태를 띈다.
- 이 때, subj-type과 obj-type은 NER에 의해서 labeling된다.

## 4. Experiments
- 제안된 기술들을 RE benchmark에 기반하여 평가한다.
- 평가의 경우, **가장 성능이 좋은 entity representation technique을 먼저 식별**한 후, SOTA 모델과의 비교를 위하여 Improved RE baseline에 추가로 통합한다.
- 더불어, **unseen entity에 대한 일반화로 이끌어주는 technique들에 대한 연구**와 **annotation error 하에서의 성능에 대한 연구**를 수행한다.

### 4-1. Preliminaries
#### Datasets
![3](https://user-images.githubusercontent.com/53552847/154454095-e324a42f-e9c3-4e95-8dcf-7684864088e1.png)
- 실험에서 original TACRED, TACREV, Re-TACRED datasets을 사용한다.
- Alt et al(2020)에서는 TACRED dataset이 6.62%의 noisily-labeld instance를 포함하고 있는 것을 발견하옸고 이를 development와 test set에 대하여 relabel했다.
- Stoica et al(2021)은 TACRED에서 정의된 label에 대한 정의를 재정의하였으며 전체 dataset에 대하여 relabel하였다.

#### Compared Methods
- PA-LSTM
    - text를 embedding으로 encoding할 때, bi-directional LSTM과 postion-aware attention을 적용한다.
    - 그 후, relation prediction을 위한 softmax layer에 넣는다.
- C-GCN
    - Graph-based model이며, 문장의 purning된 dependency tree를 graph convolutional network에  넣고 이로서 entity에 대한 representation을 얻는다.
- SpanBERT
    - Transformer based Pre-Trained language model을 사용한다.
    - Training Objective를 span prediction으로 통합하여 BERT를 확장하며 이로서 RE task에서 개선된 성능을 달성한다.
- KnowBERT
    - language model과 entity linker를 jointly하게 학습한다.
    - subtoken이 지식 기반으로 사전 학습된 entity embedding에 기여할 수 있도록 한다.
- LUKE
    - large text corpora와 knowledge graph를 둘다 language model에 사전학습한다.
    - frequent entity를 vocabulary에 더하고 entity-aware self-attention mechanism을 제안한다.
    - RE와 NER과 같은 다양한 entity-related task에서 SOTA를 달성한다.

#### Model Configurations
- Optimizer: Adam
- Learning Rate: 5e-5(BERT_BASE), 3e-5(BERT_LARGE, RoBERTa_LARGE)
- Scheduler: 첫 10%에서 linear warm-up을 진행한 후 weight decay를 0으로 초기화
- Batch_Size: 64
- Epoch: 5
- Best model의 checkpoint는 Development set의 F1 score를 기준으로 선택한다.
- 모든 실험에서 random seed를 바탕으로 5번 진행한 후 5번의 평균 F1을 최종 결과값으로 활용한다.

### 4-2. Analysis on Entity Representation
![4](https://user-images.githubusercontent.com/53552847/154454097-1ee97d18-df9d-4968-9b0d-8e049b52e3ba.png)
- BERT_BASE, BERT_LARGE, RoBERTa_LARGE를 사용한다.
- **typed entity marker와 typed entity marker (punct)의 경우, untyped entity representation보다 훨씬 성능이 좋다.**
- RoBERTa Model의 경우 typed entity marker (punct)를 사용하여 74.6%의 F1 score를 달성하며, 이는 이전의 SOTA인 LUKE의 72.7%에 비해 상당히 높은 수치이다.
- 이로서, **entity 정보의 모든 category들의 representing은 RE task에서 도움을 준다는 것을 볼 수 있다.**
- 또한, input에서 entity name을 유지하는 것이 relation classifier의 성능의 개선으로 이어지며 Zhang et al(2017)의 주장과는 모순된다.
- **entity representation 기술의 original version과 punctuation version이 entity 정보의 같은 categories를 나타냄에도 불구하고 서로 다른 성능으로 이어지며, 이로써 새로운 special token을 도입하는 것은 RoBERTa에서 성능 개선에 방해가 됨을 알 수 있다.**

### 4-3. Comparison with Prior Methods
![5](https://user-images.githubusercontent.com/53552847/154454102-b755681c-104e-42d5-999f-e99c44818461.png)
- 위의 연구들을 바탕으로, **typed entity marker (punct)와 RoBERTa_LARGE를 사용하는 것이 relation classifier에서 가장 성능이 좋다는 것을 식별**해냈다.
- 이를 바탕으로, 사전에 연구된 결과들과 비교한다.
- TACRED, TACREV, Re-TACRED에서 모든 방법들을 평가하며, 그 결과 typed entity marker (punct)와 RoBERTa_LARGE를 통합한 Improved baseline model이 모든 dataset에서 가장 성능이 좋았다.
- 추가적으로, **type entity marker를 사용함으로서 얻어지는 이득이 TACRED와 TACREV에 비해 Re-TACRED Dataset에서 상대적으로 적음을 볼 수 있었는데, 이로부터 Dataset의 높은 noise rate가 기여했다고 볼 수 있으며, noisy label들은 entity의 side information에 많이 편향된다고 볼 수 있다.**

### 4-4. Analysis on Unseen Entities
- 몇몇 Prior Work에서는 entity name이 relation type의 피상적인 정보를 누락할 수 있고 heuristic으로 인해서 benchmark에 대하여 안좋은 결과로 이어질 수 있게 한다.
    - Neural Relation Classifier가 original sentence에 subject와 object entity name을 넣지 않고도 단지 subject와 object entity name만 활용하여 높은 결과를 달성할 수 있다는 것을 보여준다.
    - 더불어, entity mask가 없이 학습된 relation classifier는 unseen entity에 대하여 잘 일반화되지 않는다고 주장한다.
- 하지만, RE dataset에서 제공된 NER type의 경우, coarse-grained하며, entity mask를 사용하는 것은 entity에 있는 의미있는 정보를 잃을 수도 있다.
    - entity mask를 사용하는 것은 relation classifier에서 entity knowledge를 넣어 사용하는 것에 대한 연구와도 상반된다.
    - 만약 relation classifier가 entity name을 고려하지 않게된다면, relation classifier가 external knowledge graph로 인해서 개선될 수 있다는 주장은 불합리하다.
- entity mask가 없이 학습된 relation classifier가 unseen entity에 대하여 일반화가 잘 되는지 평가하기 위하여 filtered evaluation setting을 제안한다.
    - TACRED, TACREV, Re-TACRED의 training set에 존재하는 entity를 포함하는 test instance를 모두 지운다.
    - 그 결과, TACRED, TACREV의 4599개의 instance의 filtered test set과 Re-TACRED에서 3815개의 instance의 filtered test set을 가진다.
    
![6](https://user-images.githubusercontent.com/53552847/154454887-886282af-ba09-4327-9b71-36c5f49bad00.png)
- entity mask와 typed entity marker representation과 함께 모델을 비교하며, entity representation에서 entity name을 포함하는지 아닌지에 대해서만 차이가 있다.
- original test set과 filtered test set에서의 label 분포는 다르며, 그들의 결과를 직접적으로 비교할 수는 없다.
- 그 결과, **여전히 typed entity marker가 모든 dataset에서 entity mask를 능가하며, 이로부터 relation classifier가 entity name으로부터 잘 학습되어질 수 있고, unseen entity에 대하여 일반화되어질 수 있음을 보여준다.**
- 이 발견은 relation classifier를 개선하기 위해서 entity type보다는 entity name에서 semantically한 rich information을 포함한다는 Peng et al(2020)의 제안과 일치한다.

### 4-5. Analysis on Annotation Errors
- Improved Baseline Model은 TACRED와 TACREV에 비해 Re-TACRED에서 상대적으로 적은 성능의 개선을 얻는다.
    - 이러한 차이가 evaluation set에서의 annotation error로부터 발생했다는 것을 발견했다.
    - 특히, TACRED test set에서 모든 instance를 제거한 clean TACRED test set을 제작하였다. (이 때, annotated relation은 Re-TACRED test set과는 다르다)
    - 남아있는 Instance는 모두 clean하다고 간주한다.
    - TACRED와 Re-TACRED에서의 label set은 다르며, 몇몇 class에서의 instance는 Re-TACRED에 없다. 즉 완전히 제거되었다.
    
![7](https://user-images.githubusercontent.com/53552847/154454890-3cd59525-5757-461d-8886-d0788721f876.png)
- 위를 바탕으로, original training set에 대하여 Model을 학습한 후, clean test set에서의 결과를 보여준다. 
- clean TACRED test set에서도 유사한 성능 향상이 관찰된다.
- 이로서, **TACRED와 TACREV에서의 annotation error가 entity의 side information에 의존하게 만들며, 이로인해 overestimation으로 이어질 수 있다는 것을 보여준다.**
- 본 과정에서, **data annotation을 할 때, 몇몇 anotator들이 전체 문장을 읽지 않고 두 entity 사이의 관계만을 기준으로 labeling할 때 많은 noise가 생성될 수 있다고 가정하며, 이로 인해 entity representation에서의 NER type을 통합하는 것이 많은 성능의 이득으로 이어진다고 볼 수 있다.**
- 전반적으로 TACRED와 TACREV에서의 evaluation set이 편향되어 있고 합리적이지 않다는 것을 보여주며, sentence-level RE에서 future work에서는 Re-TACRED를 evaluation benchmark로 사용하는 것을 추천한다.

## 5. Conclusion
- 간단하지만 강한 RE baseline를 활용하여 새로운 SOTA 성능을 이루어낸다.
- sentence-level RE에서의 두 가지 technical 문제인 "ENTITY REPRESENTATION"과 "NOISY OR ILL-DEFINED LABEL"에 대하여 재조사한다.
- **Improved Entity Representation Technique을 제공하며 이를 활용하여 존재하는 사전 relation classifier들을 상당히 능가한다.**
- **TACRED와 TACREV evaluation set이 entity information에 편향되어 있다는 것을 보여준다.**
