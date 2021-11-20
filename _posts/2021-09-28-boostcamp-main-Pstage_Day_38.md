---
layout: post
title: "Second P-stage 2(Day 38)"
categories: boostcamp
tags: main
comments: true
---
BERT 모델에 대하여 이해하고 모델 학습에 대하여 알아본다.

**부스트 캠프 38일차 학습 요약**
- **학습** : BERT 언어모델 소개, BERT 모델 학습
- **P-Stage** : 강의 실습 진행
- **피어 세션**

## 목차
- [1. BERT 모델 소개](#1-bert-모델-소개)
- [2. 한국어 BERT 모델](2-한국어-bert-모델)
- [3. BERT 모델 학습](#3-bert-모델-학습)
- [4. 해야할 일](#4-해야할-일)

## 1. BERT 모델 소개
### 1-1. AutoEncoder
- Encoder의 역할은 어떤 입력된 이미지를 압축된 데이터, 압축된 형태로 표현하는 것이 목적이다.
- Decoder의 목적은 원본을 그대로 복원하는 것이 목적이다.
- 입력이 나 자신이고 Network를 통해 어떤 벡터값이 나온다. 그 벡터값을 Decoding해서 본인 스스로가 나오도록 학습한다.
- AutoEncoder Network에서 Compressed Data Vector를 가져오면 이것이 입력된 이미지에 대한 Vector가 될 수 있다. 왜냐하면 AutoEncoder가 본인을 잘 표현하도록 학습하는 모델이기 때문이다.

### 1-2. BERT
- 위의 AutoEncoder를 그대로 대입한다.
- Self-Attention을 사용한 모델이다. 즉, input을 다시 input으로 representation하는 것을 목적으로 학습한다.
- 여기서 차이점은 주어진 문장에 Mask을 사용하는 것이다.
- input에서 그대로 input이 나오도록 학습하는 것이 아니라, mask를 더한 input에서 원본 input이 나오도록 학습함으로서 더 어려운 문제를 해결할 수 있도록 학습한다.

### 1-3. 간단한 역사도
- 순서는 아래의 순서대로 발표되었으며, 각각의 input, output은 다음과 같다.
- GPT-1 : Input -> Transformer Decoder -> Label
- BERT : Input -> Transformer Encoder -> Input
- GPT-2 : Input -> Transformer Deocder -> Input + Next

### 1-4. 모델 구조도
![11](https://user-images.githubusercontent.com/53552847/135699142-c07af59d-1270-4fbe-b34b-a32fddcdf817.PNG)
- Sentence 2개를 [SEP] token을 사이에 두고 입력으로 받는다.
- Transformer Layer가 총 12개로 구성되어 있다.
- [CLS] token이 token vector에 입력된 sentence에 대한 정보가 모두 녹아서 Embedding 될 것이다.
- Sentence 1, 2를 잘 표현하기 위해서 [CLS] token 위에 classification layer를 부착해서 pre-training을 진행한다.

### 1-5. 학습 데이터 and Tokenizing
#### 학습 데이터
- BooksCorpus (800M words)
- English Wikipedia (2,500M words without lists, tables and headers)
- 30,000 token vocabulary

#### Tokenizing
- WordPiece Tokenizing
- 입력 문장을 tokenizing하고, 그 token들로 'token sequence를 만들어 학습에 사용.
- 2개의 token sequence가 학습에 사용. 첫번째 문장을 선택하고 실제로 다음 문장 절반, 랜덤으로 선택된 문장 절반으로 두 번째 문장을 선택하여 tokenizing된다.

### 1-6. Masked Language Model
![12](https://user-images.githubusercontent.com/53552847/135699143-4d83ea4f-aecc-463e-9a26-ffdb42158b19.PNG)
- 위의 그림처럼, 최종으로 가장 아래에 있는 문장이 모델이 Input으로 들어가게 된다.

### 1-7. NLP Task(with GLUE Datset)
- MNLI : 두 문장의 관계 분류를 위한 데이터 셋
- QQP : 두 질문이 의미상 같은지 다른지 분류하기 위한 데이터 셋
- QNLI : 질의응답 데이터 셋
- SST-2 : 영화 리뷰 문장에 관한 감성 분석을 위한 데이터 셋
- CoLA : 문법적으로 맞는 문장인지 틀린 문장인지 분류를 위한 데이터 셋
- STS-B : 뉴스 헤드라인과 사람이 만든 paraphrasing 문장이 의미상 같은 문장인지 비교를 위한 데이터 셋
- MRPC : 뉴스의 내용과 사람이 만든 문장이 의미상 같은 문장인지 비교를 위한 데이터 셋
- RTE : MNLI와 유사하나, 상대적으로 훨씬 적은 학습 데이터 셋
- WNLI : 문장 분류 데이터 셋
- SQuAD v1.1/v2.0 : 질의응답 데이터 셋
- CoNLL : 개체명 분류 데이터 셋
- SWG : 현재 문장 다음에 이어질 자연스러운 문장을 선택하기 위한 데이터 셋

### 1-8. BERT를 활용한 여러 task 학습 및 표현
![13](https://user-images.githubusercontent.com/53552847/135699144-436f651a-9ea7-48ee-899b-4d26d2aab47a.PNG)
- BERT를 활용하면 위의 4가지 분류 task를 활용하며 이전에 소개한 모든 NLP task를 수행할 수 있다.
- 단일 문장 분류
    - BERT 모델에 한 개의 문장이 입력되었을 때, 문장이 어떤 class에 속하는 지를 분류하는 task
- 두 문장 관계 분류
    - 두 개의 문장이 입력되며, Next Sentence, 인과관계, 유사도 등을 비교하는 task를 수행할 수 있다.
- 문장 토큰 분류
    - [CLS] token에 입력된 sentence에 대한 정보가 녹아들어있다라는 가정을 둔다.
    - 각 token들의 output에 분류기를 부착하여, 각 token이 어떤 label을 가지는지 분류한다.
- 기계 독해 정답 분류
    - 두 정보가 주어진다. (질문, 정답이 포함된 문서)
    - 정답이 포함된 문서 내에서 질문이 대한 정답이 어디에 있는지 알아낸다.
    - 즉, 정답의 start point와 end point가 어디인지 알아내는 task이다.

#### BERT의 핵심(with task)
- 동일한 모델을 활용하여 모든 task를 진행한다.
- Input Setence만 바뀐다. 즉, tokenizer의 역할이 중요하다.
- Classifier는 거의 동일하게 진행된다.
- 위의 4개의 분류 방법을 사용하면, 거의 모든 자연어 처리 task를 다 해결할 수 있다.
- 각각의 task에 대한 코드 차이가 크지 않다.

## 2. 한국어 BERT 모델
- ETRI KoBERT
    - WordPiece Tokenizer를 사용 하되, 바로 사용하는 것이 아니라, 먼저 형태소 단위로 분리를 한 후에 WordPiece Tokenizer를 거친다.
    - 형태소의 최소역할은 의미를 가지는 최소단위로 분리하겠다는 것인데, 형태소 단위로 분리한 후에 WordPiece Tokenizer를 사용하게 되면 한국어 형태에 더 알맞은 tokenizer 형식이 될 수 있다.
    - ETRI 형태소 분석기를 사용해야만 ETRI KoBERT를 사용할 수 있다는 단점이 있다.
- SKT KoBERT
    - 형태소 tokenizer를 사용하지 않는다.
    - WordPiece Tokenizer만 사용한다.
    - KorQuAD에서는 ETRI KoBERT 성능보다는 낮다.
- Advanced BERT Model
    - KorQuAD에서의 성능을 더 내기 위한 실험으로서, 기존의 WordPiece, Segment, Positional Embedding만 넣는 것이 아닌, Entity에 대한 내용을 추가하였다.
    - 정답을 내기 위한 feature는 당연히 entity가 될 것이고, BERT 내에서는 entity를 명시할 수 있는 구조가 존재하지 않기 때문에, 원본 문장에 entity tag를 추가해주고 entity tag에 대한 embedding을 추가하였더니 성능이 더 증가하였다.
![14](https://user-images.githubusercontent.com/53552847/135699145-4f93e9ff-e258-4a78-9a38-8caec62b728e.PNG)
    - 이러한 사실들을 바탕으로, 언어 모델을 학습할 때, feature가 무엇이 될 것인지 고민하고, 사람이 자연어 처리 task를 할 때는 어떤 feature를 사용할 지에 대한 고민을 해보고 이를 모델에 함께 녹이는 방법들을 고민해보는 것이 좋다.

## 3. BERT 모델 학습
### 3-1. BERT 학습의 단계
- Tokenizer 만들기
- Dataset 확보하기
- Next Sentence Prediction(NLP)
- Masking

### 3-2. Mixed-Domain Pretraining
![15](https://user-images.githubusercontent.com/53552847/135699146-57c3088e-7a6b-4c4d-904a-f9ef2430daa5.PNG)
- 도메인 특화 task의 경우, 도메인 특화된 학습 데이터만 사용하는 것이 성능이 더 좋다.
- fine-tuning 하는 것보다, 바닥부터 다시 학습한 모델이 성능이 더 좋음을 알 수 있다.
- 즉, 해당 도메인에 관련된 corpus를 가지고 학습을 하는 것이 더 좋은 성능을 나타낼 수 있다.
- 방법
    - Dataset 만들기
        - input_ids : token 단위로 vocab id로 전달
        - token_type_ids : segment Embeddings
        - Position Embeddings
    - DataLoader 만들기
        - Masking : Dataset 중간중간에 Mask가 되어있는 데이터를 모델의 입력으로 넣어준다. 
        
### 3-3. Pre-Training시 유의사항
- 개인정보는 반드시 삭제한 후에 pre-training 진행한다.
- 개인정보가 있는 채로 pre-training을 진행할 경우, [MASK]를 예측하는 형태로 학습하게 되는데, [MASK]를 예측하는 과정에서 개인정보를 활용해 pre-training되기 떄문에, 개인정보를 바탕으로 한 예측값이 출력될 수 있다.

## 4. 해야할 일
- Embedding이 잘 되었는지, 안되었는지 평가할 수 있는 방법은 무엇이 있을까?
    - WordSim353
    - Spearman's Correlation
    - Analogy Test
- Vanilar Transformer는 어떤 문제가 있고 어떻게 극복할까?
    - Longformer
    - Linformer
    - Reformer
- 텍스트 정제는 정말 필요할까?
    - 실제로 우리가 웹이나 메신저를 통해 사용하는 언어는 '정제되지 않은 언어'이다.
    - 해당 데이터가 적용되는 방향에 따라서 정제가 필요할 수도, 필요하지 않을 수도 있다.
    - 오히려 더욱 어려운 데이터로 학습한 모델의 성능이 더 좋을 수 있다.
