---
layout: post
title: "[논문리뷰] Dense Passage Retrieval for Open-Domain Question Answering"
categories: booststudy
tags: paper
comments: true
---
Dense Passage Retrieval for Open-Domain Question Answering을 읽고 이에 대하여 논의한다.

본 내용은 [DPR_ODQA_Paper](https://arxiv.org/pdf/2004.04906.pdf)을 바탕으로 작성했다.

## 목차
- [1. Abstract](#1-abstract)
- [2. Introduction](#2-introduction)
- [3. Dense Passage Retriever](#3-dense_passage_retriever)
- [4. Experiments: Passage Retrieval](#4-experiments-passage-retrieval)
- [5. Experiments: Question Answering](#5-experiments-question-answering)
- [6. Related Work](#6-related-work)
- [7. 해야할 일](#7-해야할-일)


## 1. Abstract
- ODAQ(Open-domain Questino Answering)은 효율적은 Passage Retrieval에 의존한다.
- 전통적으로 TF-IDF/BM25와 같은 Sparse Vector space Model을 활용하였다.
- 본 논문에서는 간단한 Dual Encoder(두 개의 Encoder)로 작동하고, 작은 수의 Q(question), P(Passage)로부터 학습된 Dense Representation를 보여준다.
- 본 논문에서 소개한 Dense Retrieval는 top-20 Passage에 대하여 BM25대비 9~19%의 성능이 증가함을 보여준다.
- [Multiple ODQA benchmarks](https://github.com/facebookresearch/DPR)에서 SOTA 달성

## 2. Introduction
### 2-1. Dense Representaion
- **Dense Representation은 Sparse Representation에 비해 synoyms/paraphrases를 더 잘 mapping**한다.
    - 예를 들어, {Q: "Who is the bad guy in lord of the rings?", P(ground_truth): "Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy."}가 있다고 할 때, **Term-based system(Sparse Embedding)은 'bad guy'를 'villain'과 매칭시키지 못해 올바른 Passage를 Retrieve하기 어려운 반면, Dense Retrieval은 더 잘 매칭시킨다.**
- **Dense Representation**은 학습이 가능하며, 이로 추가적으로 task-specific하게 만들 수 있다.  

### 2-2. Why do we need to use this Dense Retrieval(본 논문에서 소개한 Retrieval)
- Good Dense Vector Representation을 위해서는 방대한 양의 labeling된 (Question, Passage) pair가 필요하다고 믿어져왔다.
- ORQA가 등장하기 전에는 TF-IDF/BM25를 능가하는 Dense Retrieval가 없었다.
- 하지만 ORQA 역시 2가지의 약점을 가지고 있으며, 본 논문에서 소개한 Dense Retrieval가 Multiple ODQA datasets에서 SOTA를 달성하였다.
    - ORQA의 경우, ICT(inverse cloze task) pretraining을 활용하였는데, 이는 computationally intensive하며 regular sentence가 좋은지에 대해서는 명확하지 않다.
    - Passage Encoder를 fine-tuning하지 않기 때문에, 이는 suboptimal이 될 수 있다.
    - ORQA에 대한 보다 자세한 사항은 [ORQA Paper](https://arxiv.org/pdf/1906.00300.pdf)에서 확인할 수 있다.

### 2-3. DPR introduced in this paper.
- '추가적인 pretraining없이 (Q, P) pair만을 이용해 더 나은 Dense Embedding Model을 학습시킬 수 있을까?' 라는 질문 아래 시작됬다.
- Retrieval에서의 Backbone Model은 'BERT pretrained model'을 활용하였다.
- Dual Enocder Architecture를 활용하였다. (Question Encoder, Passage Encoder)
- **Retrieval의 최종 Embedding vector는 Question vector와 Relevant Passage vector 사이의 내적을 최대화함으로서 최적화 되었다.**
- **더불어, Batch 내에서의 모든 (Q, P) pair들을 활용하여 대하여 학습되었다.**
- Top-5 Accuracy에서 BM25에 대비 훨씬 성능이 좋아졌으며(42.9 -> 65.2), End-to-end QA Accuracy 역시 (33.3 -> 41.5)로 훨씬 개선되었다.
- 핵심적인 요소는 **Question Encoder와 Passage Encoder를 모두 fine-tuning 했다는 것**이고 이만으로도 BM25를 크게 능가하였다.
- 본 논문에서의 경험적인 결과로서 **추가적인 pretraining은 필요하지 않을 것**이라고 제안하였다.
- 더불어, **좋은 Retriever를 사용하는 것이 좋은 ODQA Accuracy로 이어짐을 보였다.** 

## 3. Dense Passage Retriever
- Dense Retrieval을 활용하여 질문과 관련된 상위 k개의 연관된 Passage를 효율적으로 제공할 수 있도록 한다.
- k는 보통 작은 수이며, 흔히 20~100을 사용한다.

### 3-1. Overview
- **Passage Encoder**로서 $$E_p$$를 활용하며 모든 passage들을 d차원을 가지는 real-valued vector로 mapping하며 모든 Passage들에 대하여 index를 build한다.
- **Question Encoder**로서 $$E_q$$를 활용하며 이는 input question을 d차원 vector로 mapping한다.
- 단순히, **DP(dot product)를 활용하여 question과 passage 사이의 유사도**를 정의한다.
![1](https://user-images.githubusercontent.com/53552847/137823027-ca466763-05cb-439a-add0-912957f98401.PNG)


#### Why do we use DP?
- DP(dot product)와 NLL(Negative Log-likelyhood Loss)에 더불어, Euclidean distance(L2)와 triplet loss에 대해서 역시 실험을 진행했다.
- 다음의 표와 같은 결과가 나왔다.
![2](https://user-images.githubusercontent.com/53552847/137823028-f4c88490-fdce-4dd0-b612-ce2556b979d8.PNG)
- **이러한 이유로, DP(dot product) 유사도를 활용**하였고, 이로부터 더 나은 better encoder를 학습할 수 있었다.

#### Encoder
- **Question/Passage에 대하여 독립적인 BERT Encoder**를 활용하였다.(BERT-base-uncased)
- **[CLS] token에 대한 hidden embedding vector를 representation**으로 활용하였다.
- **hidden dimension size는 768**로 지정하였다.

#### Inference
- Passage Encoder $$E_p$$에 대하여 FAISS를 적용하였다.
- FAISS는 효율적으로 similarity search를 진행할 수 있으며, indexing을 하는데 있어 도움을 주는 open source library이다. FAISS에 대한 자세한 내용은 [Boostcamp-FAISS](https://jjonhwa.github.io/boostcamp/2021/10/15/boostcamp-main-Pstage_Day_49/)에서 확인할 수 있다.

### 3-2. Training
- 학습의 목표는 Question vector와 Passage vector 사이의 거리를 관계가 있는 것은 가깝게 관계가 없는 것은 멀게 만드는 것이다.
- 학습 데이터의 모양은 다음과 같으며, 여기서 q는 Question, p+는 positive passage, p-는 negative passage를 의미한다.
![3](https://user-images.githubusercontent.com/53552847/137823029-1789fb72-121d-4960-aeba-22d2a7d422bf.PNG)
- **Loss Function은 NLL을 활용**하였으며 식은 다음과 같다.
![4](https://user-images.githubusercontent.com/53552847/137823030-2bf8fbf8-1be6-4d33-8015-f32240f44d00.PNG)


#### Positive and Negative passages
- 보통 positive passage들은 명확히 선택할 수 있는데에 반해 negative passage들은 많은 passage들 중에서 다양한 방법으로 선택될 수 있다.
- **negative passage를 선택하는 것은 간단하지만 좋은 encoder를 학습시키는데 중요한 요소**가 될 수 있다.
- Different types of negatives
    - Random : corpus에서 랜덤으로 선택
    - BM25 : question에 대한 정답을 포함하고 있지 않으면서 BM25기반 question과 높은 유사도를 가지고 있는 passage를 선택
    - Gold : training set에서 나타난 다른 질문에 대한 positive passage를 선택.
- 본 논문에서는 **각 question 당 같은 mini-batch 안에서의 gold passage들과 BM25기반 가장 높은 유사도를 가지며 정답을 포함하고 있지 않은 passage를 negative passage로 선택하여 학습**한다. (이후 과정에서 자세히 확인할 수 있다.)
- **같은 batch 내에서 gold passage를 negative passage로서 활용하는 것은 계산적인 효율뿐만 아니라 좋은 성능을 내는데 도움이 된다.**(In-batch negatives에서 자세히 확인할 수 있다.)

#### In-batch negatives
mini-batch내의 B개의 Question이 있다라고 가정할 때, 각 Question들은 관련된 passage를과의 연관성을 고려한다. Q, P가 각각 Question Matrix, Passage Matrix(Bxd)일 때(B: # of questions, d: hidden_dimension_size), 유사도는 Q와 P의 내적을 이용해 구할 수 있고 그 결과 $$S = QP^T$$ (BxB)가 된다. S의 각 행은 B개의 passage들과 쌍을 이루는 question에 대응한다.
- 즉, S는 batch내에서 question와 passage 사이의 유사도를 담고 있는 행렬이다.
- 그 결과, S의 대각원소는 question과 postiive passage 사이의 유사도를 나타내고 비대각원소는 negative passage들과의 유사도를 나타낸다.
- 이렇게 함으로서, **negative sampling을 할 때의 계산 비용이 in-batch를 하지 않을 때에 비해 비약적으로 줄어든다.**(batch내에서의 similarity Matrix를 만들고 이로부터 바로바로 재사용할 수 있기 때문이다.)
- 더불어, **성능 또한 개선된다.**(이후 과정에서 자세히 확인할 수 있다.)
- 이는 dual-encoder model을 학습하는 데 있어서 효과적인 전략임을 보여주었다.

## 4. Experiments: Passage Retrieval
### 4-1. 학습
- **in-batch negative setting** (같은 batch 내에서의 negative sampling 적용)
- **batch_size : 128**
- **각 question별로 batch 내에서의 BM25기반 가장 유사도가 높으면서 정답을 포함하지 않은 passage를 negative passage로 활용.**
- **40 epoch (NQ, TriviaQA, SQuAD)**
- **100 epoch (TREC, WQ)**
- **learning rate : $$10^{-5}$$**
- **optimizer : Adam**
- **scheduler : linaer scheduling with warm-up**
- **dropout ratio : 0.1**
- 이와 더불어, BM25기반, DPR기반, BM25 + DPR기반을 활용하여 비교분석을 진행하였다.
- BM25 : b = 0.4 / $$k_1$$ = 0.9 활용
- **BM25 + DPR**
    - new ranking function을 활용하여 linear combination을 진행
    - **BM25/DPR에서 top-2000개의 passage를 뽑은 후, 다음의 식을 활용하여 top-k개의 최종 passage를 선택**
![5](https://user-images.githubusercontent.com/53552847/137823033-17c34eb5-6fe3-4059-a6f0-74c787a4c01f.PNG)
    - 이 때, **lambda는 가중치 상수이며 본 논문에서는 Retrieval 정확도에 기반하여 1.1을 활용**하였다.

### 4-2. Main Results
![6](https://user-images.githubusercontent.com/53552847/137823008-7d78b45a-8aef-44f5-a497-d904bce5f75b.PNG)
- 위의 표처럼, top-20/top-50에 대하여 5개의 QA datasets에 대한 실험을 진행하였다.
- SQuAD를 제외하고는 DPR의 성능이 BM25보다 뛰어나는 것을 확인할 수 있다.
- Single의 경우, 각각의 Dataset에 대하여 학습시킨 DPR을 활요한 것이며, Multiple의 경우 SQuAD를 제외한 4개의 Dataset을 합쳐서 학습시킨 DPR을 활용한 결과이다.
- 상대적으로 작은 dataset인 TREC, WQ의 경우 Multiple 학습을 진행하였을 때, 상당히 눈에띄는 성능 개선을 보여준다.
- 그에 반해, 큰 dataset에서는 Multiple을 활용했을 때 약간만 개선된 모습을 보여준다.

#### Why does BM25 perform better in SQuAD dataset?
- annotation bias
    - 데이터셋을 제작할 때 생긴 bias 때문이라고 생각해볼 수 있다.
    - SQuAD dataset의 경우, passage를 제공받은 후 본 passage 내에서 답을 가질 수 있는 질문을 만드는 과정으로 만들어지는데, 이 때 passage에 dependent하게 question을 만들 확률이 높다.
    - 이로인해, question의 token들이 passage에 포함되어 있을 확률이 높고, 이로 인해 DPR보다 BM25에서 성능이 더 좋다고 추측할 수 있다.
- biased examples
    - SQuAD Dataset은 사람들이 많이 보는 문서 약 500개로부터 만들어진 passage들로부터 가공된 dataset이기 때문에, 이에 한정하여 많은 bias가 있음을 추측할 수 있다.

### 4-3. Ablation Study on Model Training
#### Sample efficiency
![7](https://user-images.githubusercontent.com/53552847/137823015-3e442390-87c9-413c-a07d-43516eb868f9.PNG)
- **좀 더 많은 training dataset을 활용할 수록 retrieval accuracy는 증가**한다.
- top-k 즉, **retrieval할 최종 passage의 개수를 늘릴 수록 retrieval accurayc는 증가**한다.

#### In-batch negative training
![8](https://user-images.githubusercontent.com/53552847/137823020-77e4d628-71a3-48a2-b2f9-97396c69b79e.PNG)
- #N : Negative sample의 개수를 의미한다.
- IB : In-Batch, 즉 batch 내에서의 negative sampling을 진행함을 의미한다.
- Top-k : 최종 k개의 passage 내에서의 ground-truth passage가 있을 확률을 의미한다.
- 위의 표에서 **Random/BM25/Gold 사이의 차이를 확인하면, k가 20보다 커질 경우에는 큰 차이가 없음**을 확인할 수 있다.
- yellow color : In-Batch, 즉 batch 내에서의 negative sampling을 하냐 안하냐에 따른 차이를 보여주며, **In-Batch Negative Sampling이 확실한 성능 개선이 있음을 보여준다.**
- pink color : batch_size에 따른 차이를 보여주며, **batch_size가 커짐에 따라 성능이 개선이 있음을 보여준다.**
- red color : BM25를 활용한 hard negative passage(유사하지만 정답을 포함하지 않는 passage)를 추가적으로 같은 batch내에서 활용함에 따른 차이를 보여주며, **BM25 negative passage를 활용하는 것이 성능의 개선으로 이어짐을 보여준다.**
- **더불어, BM25 negative passage를 각 question별 1개를 사용하는 것과 2개를 활용하는 것은 큰 차이가 없음을 확인할 수 있다. 즉, 2개 이상을 활용하는 것은 더이상 도움이 안된다고 판단한다.**

#### Impact of gold passages
![9](https://user-images.githubusercontent.com/53552847/137823022-d52a3947-feec-4ada-b157-4b57e6f21150.PNG)
- 본 논문에서는 dataset 내에서의 gold context(question에 대한 ground-truth passage)를 활용하여 학습을 진행한다.
- 이에 대하여, 위의 표에서 처럼 **Dist. Sup과 Gold 사이의 차이를 실험하여 진행하였으며 Gold Context를 활용하였을 때 보다 성능이 좋음**을 볼 수 있다.
    - Dist. Sup : BM25를 활용하여 context들 중에서 정답을 포함하면서 가장 확률이 높은 context를 ground-truth passage로 활용한다. 

#### Similarity and Loss
![10](https://user-images.githubusercontent.com/53552847/137823023-53a6fb06-fe0b-478b-b814-2423953b96e0.PNG)
- L2 norm은 DP(dot product)와 비슷한 성능을 내며, 이 둘 모두 cosine 유사도보다 높은 성능을 낸다.
- negative log-likelihood와 더불어 triplet loss를 활용하여 비교 실험을 진행하였다.
- 위의 표에서 처럼, 최종적으로 **DP와 NLL을 활용하였을 때, 가장 좋은 성능**을 냈고 이에 따라 본 논문에서 역시 DP와 NLL을 활용하여 진행하였다.

#### Cross-dataset generalization
- '추가적인 fine-tuning없이 다른 dataset에 대해서도 잘 작동하는가?'에 대한 것을 확인해본다.
- 본 논문에서는, NQ(Natural Question) dataset에 대해서만 DPR를 학습시킨 후, WQ(WebQuestions), TREC(CuratedTREC)에 실험을 해본 결과, 상당히 잘 작동함을 확인하였다.
- top-20 accuracy : WQ(75% -> 69.9%) / TREC(89.1% -> 86.3%) (Multiple 대비 NQ에서만 학습했을 때의 성능이다.)
    - 이 수치는 BM25(WQ-55.0/TREC-70.9)에 비하면 충분히 유의미하다고 볼 수 있다.

### 4-4. Qualitative Analysis
- Term-matching method(Sparse Embedding)
    - BM25와 같은 Term-matching method의 경우, 매우 선택적인 keyword나 구절에 대해서는 매우 민감하게 반응한다.
- DPR
    -  어휘적 변화나 의미적 관계를 더 잘 포착한다.

### 4-5. Run-time Efficiency
- ODQA를 수행함에 있어, 사용자가 질문에 대한 답을 실시간으로 얻는 과정에서 reader model에서 답을 찾기 위한 passage들의 개수를 줄이는 것은 핵심 포인트이다. -> 이에 따라, passage retrieval 속도를 확인해보도록 한다.
- CPU : Intel Xeon CPU E5-2698 v4 @ 2.20GHz and 512GB memory
- DPR
    -  초당 995개의 질문을 처리하고, 각 질문당 top-100개의 passages를 반환한다.
    -  반면, **building an index for dense vector 과정에서 21백만개의 passage를 처리하는데 8개의 GPU로 병렬처리하여 8.8시간이 걸린다.**
    -  **FAISS index를 활용할 경우, 하나의 GPU server로 8.5시간이 걸린다.**
- BM25
    - 23.7개의 질문을 처리한다. per second per CPU thread
    - building ans inverted index를 하는데 30분이 채 걸리지 않는다.

## 5. Experiments: Question Answering
### 5-1. End-to-end QA System
- Retriever로부터 주어진 top-k passage들에 대하여 reader model이 최종 answer를 추출해낸다.
- **reader는 각 passage들에 대하여 passage selection score를 부여한다.**
- **각 passage로부터 answer span을 추출하고 span score를 부여한다.**
- **가장 높은 passage selection score와 best span score를 가진 span이 최종 정답으로 선택된다.**
- **passage selection model : question과 passage 사이의 cross-attention을 통해 re-rank**한다.   
    - cross-attention은 분해할 수 없는 특성으로 인해서  대규모 말뭉치의 관련 구절을 검색할 수는 없지만 dual-encoder model의 similarity보다는 좀 더 용이하다.

#### How can you get the answer span? 
![11](https://user-images.githubusercontent.com/53552847/137823025-28c9fbb9-ed78-496f-a0df-1d2035feac0e.PNG)
- $$P_i$$는 i번쨰 passage에 대한 BERT representation이며, (Lxh)를 가지며 L은 maximum length of the passage, h는 hidden dimension을 의미한다.
- 정답 span의 처음에 위치할 확률과 마지막에 위치할 확률은 위의 그림과 같이 계산된다.
- passage가 선택될 확률 역시 위의 그림과 같이 계산된다.
    - passage가 선택될 확률은 모든 passage들의 [CLS] token에 대한 hidden embedding vector와 학습가능한 vector $$w_{selected}$$와의 연산을 통해 계산된다.
- $$P_{selected}(i)$$ 에 의해서 어떤 passage가 선택될 것인지 구한 후, ($$P_{start,i}(s)$$ x $$P_{end,i}(t)$$ )에 의해서 최종 span score가 계산된다.

#### Reader Training
- training을 하는 동안, **Retriever로 부터 주어진 100개의 passage들 중에서 1개의 positivie passage와 m-1개의 negative passage를 sampling**한다.
- **m은 hypter-parametere로서 본 논문에서는 모든 실험에서 m = 24로 활용**하였다.
- Reader에서의 학습은 **선택된 positive passage의 log-likehood와 함께 positive passage에서의 모든 정답 span의 Marginal  log-likelihood를 최대화하는 방향으로 학습**된다.(모든 정답 span이라고 한 이유는, 한 passage에서 여러 정답이 나타날 수 있기 때문이다.)
- batch_size : 16(NQ, TriviaQA, SQuAD), 4(TREC, WQ)

### 5-2. Results
![12](https://user-images.githubusercontent.com/53552847/137823026-96066d9d-5a00-4c3b-b874-703ff99f5649.PNG)
- retriever accuracy가 높을 수록 전체 ODQA 결과 역시 더 높다.
- **ORQA/REALM 모두 추가적인 pretraining task와 expensive end-to-end training regime를 가지고 있는데 반해, 본 논문에서의 DPR Retrieval를 활용한 ODQA는 간단히 (Q, P) 쌍에 대한 강력한 DPR model만을 활용하여 NQ, TriviaQA에서 더 좋은 성능을 도출**했다.
- **추가적인 pretraining task는 dataset이 작을 경우에 더 유용할 수 있다.**
- Retrieval와 Reader가 함께 학습되는 모델([joint model](https://arxiv.org/pdf/1906.00300.pdf))과의 비교 실험을 진행했으며, 이 때 39.8 EM이 나왔으며, 이에 비해 독립적인 Retriever와 Reader를 사용하는 것이 더 좋은 성능을 나타내며 더 좋은 전략임을 제안한다.
- 더불어, inference하는 과정에서 명확히 얼마나 더 많은 시간이 걸리는지 알 수는 없지만 ORQA에 비해 본 논문에서 사용한 Reader가 더 많은 Passage를 고려하여 task를 수행한다.

## 6. Related Work
- 지식 그래프와 위키피디아 하이퍼링크와 같은 외부의 구조화된 정보로 텍스트 기반 Retrieval을 하는 것이 최근에 연구되고 있다.[LEARNING TO RETRIEVE REASONING PATHS OVER WIKIPEDIA GRAPH FOR QUESTION ANSWERING](https://arxiv.org/pdf/1911.10470.pdf)
- [Poly-encoders](https://arxiv.org/pdf/1905.01969.pdf)에서 효율적인 re-ranking task를 보여주고있다.
- IR task에서의 full dense retrieval의 실현가능성을 입증하였다. [ColBERT](https://arxiv.org/pdf/2004.12832.pdf)
- Generation Model(BART, T5)과 결합한 DPR(knowledge-intensive task에서 좋은 성능을 보인다.) - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf)

## 7. 해야할 일
- ICT -> 'ICT pretraining is computationally intensive and it is not completely clear that regular sentences are good surrogates of questions in the objective function'에 대한 내용 이해하기
- reader model -> passage selection model 방법 알아보기 -> [RECONSIDER Paper](https://aclanthology.org/2021.naacl-main.100.pdf)

본 리뷰는, 'Dense Passage Retrieval for Open-Domain Question Answering'을 바탕으로 번역 및 본인 스스로의 의견을 덧붙여 작성하였으며 잘못된 내용이 있다면 의견을 남겨주시면 감사하겠습니다!
