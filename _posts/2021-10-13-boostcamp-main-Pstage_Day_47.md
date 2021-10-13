---
layout: post
title: "Third P-stage 2(Day 47)"
categories: boostcamp
tags: main
comments: true
---
Extraction-based MRC과 Generation-based MRC에 대하여 학습한다.

**부스트 캠프 47일차 학습 요약**
- **행사** : 마스터 
- **학습** : Extraction-based MRC, Generation-based MRC
- **피어 세션**

## 목차
- [1. Extraction-based MRC](#1-extraction-based-mrc)
- [2. Generation-based MRC](#2-generation-based-mrc)
- [3. Extraction-based MRC vs Generation-based MRC](#3-extraction-based-mrc-vs-generation-based-mrc)

## 1.Extraction-based MRC
- 질문과 답변이 항상 주어진 지문 내의 span으로 존재
- Extraction-based MRC의 경우 질문에 대한 답을 생성하는 것이 아닌 지문 내에서의 위치를 파악하는 것으로 문제를 해결할 수 있어서 보다 편리하다.
- ex) SQuAD, KorQuAD, NewsQA, Natural Questions, etc.
- Extraction-based MRC datastets 역시 HuggingFace의 datasets 내에 존재한다.

### 1-1. Extraction-based MRC 평가 방법
![16](https://user-images.githubusercontent.com/53552847/136954390-0e33d3b7-e255-4a39-b4d1-0af9c81a3575.PNG)
![17](https://user-images.githubusercontent.com/53552847/136954392-994fce21-79e1-4e4f-848a-de48bcd3b619.PNG)
- EM : 정확히 일치했을 경우 점수 부여
- F1 Score : 부분적으로 일치할 경우에도 일정 점수 부여
- 위의 이미지의 예제에서 볼 수 있듯이 점수가 계산된다.

### 1-2. Extracion-based MRC Overview
![18](https://user-images.githubusercontent.com/53552847/136954394-0c44858a-4348-4eef-a06e-478051a71f51.PNG)
- Context와 Question이 tokenization되어 input으로 들어가게 된다.
- Word Embedding을 통해 vector화 시킨다.
- Vector화된 값이 위의 모델의 파란색 박스에 들어가서 start position과 end position에 대한 예측값을 출력하게 된다.
- 모델을 통과하면서 바로 start position과 end position의 예측값을 출력하는 것은 아니고, conext와 question에 해당하는 contextualize vector를 출력하게 되는데 이를 각각 scaler value로 내보냄으로서 각 포지션이 시작과 끝이 될 수 있는 score를 도출해 가장 높은 socre를 찾는 방식으로 예측을 진행한다.
- 시작과 끝의 span을 가져와서 최종 답으로 내보낸다.

### 1-3. Pre-processing
#### Tokenization
- 텍스트를 작은 단위로 나누는 것
- 띄어쓰기 기준, 형태소, subword 등 여러 단위 토큰 기준이 사용된다.
- Out-of-Vocabulary (OOV) 문제를 해결해주고 정보학적 이점을 가진 Byte Pair Encoding(BPE)를 주로 사용한다.
- BPE 방법론 중 하나인 Word Piece Tokenizer를 자주 사용한다.
- Special Token을 활용하여 질문과 context를 구분한다.

#### Attention Mask
- 입력 시퀀스 중에서 attentio을 연산할 때 무시할 토큰을 표시하낟.
- 0은 무시하고 1은 연산에 포함시킨다.
- 보통 [PAD]와 같은 의미가 없는 특수 토큰을 무시하기 위해 사용한다.

#### Token type ids
- 입력이 2개 이상의 sequence일 경우(e.g. 질문 & 지문), 각각에게 ID를 부여하여 모델이 구분해서 해석할 수 있도록 유도한다.

#### 모델 출력값
- 정답은 문서 내 존재하는 연속된 단어 토큰(span)이므로, span의 시작과 끝 위치를 알면 정답을 맞출 수 있다.
- Extraction-based에서는 답안을 생성하기 보다는 시작위치와 끝위치를 예측하도록 학습함으로서 Token Classification 문제로 치환하여 해결할 수 있다.

### 1-4. Fine Tuning
![19](https://user-images.githubusercontent.com/53552847/136954398-da0c0464-add1-42f2-a5e8-ab3a026498da.PNG)
- 지문 내에서 정답에 해당하는 각 embedding을 실제로 linear transformation을 통해 각 단어마다 하나의 숫자가 나올 수 있도록 바꿔준다. 이 때, linear transformation 역시 학습 대상이다.
- 위의 BERT에서 볼 때, 각 token마다 하나의 숫자 output이 나오게 되고, 이 숫자는 점수로 볼 수 있으며, 각 token에서의 점수가 가장 높은 token이 특정 point로서 start point, end point로 학습되어 진행된다.
- 실제로 학습이 진행될 때는 softmax 위에 apply하여 negative log likelihood로 학습하는 방식을 취한다.

### 1-5. Post-preprocessing
- 불가능한 답 제거하기
    - End position이 start position보다 앞에 있는 경우 삭제
    - 예측한 위치가 context를 벗어나는 경우 삭제
    - 미리 설정한 max_answer_length 보다 길이가 긴 경우 삭제
- 최적의 답안 찾기
    - Start/End Position Prediction에서 socre(logits)가 가장 높은 N개를 각각 찾는다.
    - 불가능한 start/end 조합을 제거한다.
    - 가능한 조합들을 score의 합이 큰 순서대로 정렬한다.
    - Score가 가장 큰 조합을 최종 예측으로 선정한다.
    - Top-k가 필요한 경우 차례대로 내보낸다.

## 2. Generation-based MRC
![20](https://user-images.githubusercontent.com/53552847/136954402-213025a3-7478-4318-9273-b3b0bd0dee69.PNG)
- 주어진 지문과 질의를 보고, 답변을 생성하는 생성 문제로 접근한다.
- Extraction-based MRC 문제는 Generation-based MRC 문제로 치환할 수 없지만 그 역은 성립하지 않는다. (정답이 문장 내에 존재하지 않을 경우 역은 성립 x)
- Generation-based MRC의 경우 설사 정답이 지문 내에 존재한다 하더라도, 그 정답의 위치를 파악하는 것이 아니라 모델이 해당 정답을 생성할 수 있도록 유도하고 실제로 생성한 값을 예측값으로 사용하게 된다.

### 2-1. Generation-based MRC 평가방법
- Extraction-based MRC와 같은 평가방법을 사용할 수 있지만 생성 문제와 일반적으로 비슷하게 접근하기 위해서 ROUGE-L과 BLEU를 사용하기도 한다.

### 2-2. Generation-based MRC Overview
![21](https://user-images.githubusercontent.com/53552847/136954408-caccefc8-005e-4ae0-95e8-7c6cd636a1ea.PNG)
- input의 경우 Extraction-based model과 큰 차이가 없다.
- 위의 그림에서 볼 수 있는 Green box에서 정답까지 생성한다.
- 일종의 Seq2seq 모델로 이해할 수 있다. 모든 pre-trained 모델이 seq2seq을 할 수 있는 것은 아닌데, BERT의 경우 Encoding만 있고 Decoding이 없기 때문에 Generation-based MRC에 활용할 수 없다.

### 2-3. Pre-processing
#### Tokenizing
- 텍스트를 의미를 가진 작은 단위로 나누는 것
- WordPiece Tokenizer를 사용
    - WordPiece Tokenizer 사전 학습 단계에서 먼저 사용한 전체 데이터 집합에 대해 구축되어 있어야 한다.
    - 구축 과정에서 미리 각 단어 토큰들에 대해 순서대로 번호를 부여해둔다.
- 입력 텍스트를 토큰화한 뒤, 각 토큰을 미리 만들어 둔 단어 사전에 따라 인덱스로 변환한다.
- 인덱스로 바뀐 질문을 보통 input_ids(혹은 input_token_ids)라고 한다.
- 모델의 기본 입력은 input_ids만 필요하나, 그 외 추가적인 정보가 필요하다. (special token, attention mask, etc)

#### Special Token
![22](https://user-images.githubusercontent.com/53552847/136954333-0566cc9c-4578-40ae-8532-43783673b668.PNG)
- 학습 시에만 사용되며 단어 자체의 의미는 가지지 않은 특별한 토큰
- SOS, EOS, CLS, SEP, PAD, UNK 등등
- Extraction-based MRC에서는 CLS, SEP, PAD 등의 토큰을 사용한다.
- Generation-based MRC에서도 PAD 토큰은 사용되나, CLS, SEP의 경우 정해진 텍스트 포맷으로 데이터를 생성한다.

#### Attention Mask
- Extraction-based MRC와 똑같이 attention 연산을 수행할지 결정하는 attention mask가 존재한다.

#### Token type ids
![23](https://user-images.githubusercontent.com/53552847/136954341-9f23fcef-acd0-49aa-8bb2-b445bb3aec36.PNG)
- BERT와 달리 BART에서는 입력시퀀스에 대한 구분이 없어 token_type_ids가 존재하지 않는다.
- 따라서 Extraction-based MRC와 달리 token_type_ids가 들어가지 않는다.

#### 모델 출력값
![24](https://user-images.githubusercontent.com/53552847/136954343-bdd959bc-f5ac-4ac5-90bb-98ac5de66e1d.PNG)
- Extraction-based MRC의 경우, 텍스트를 생성해내는 대신 시작과 끝 토큰의 위치를 출력하는 것이 목표. 즉, 전체 시퀀스의 각 위치마다 모델이 아는 모든 단어들 중 하나의 단어를 맞추는 classification 문제이다.
- Generation-based MRC의 경우, 실제 텍스트를 생성하는 것이 목표
- Generation-based MRC의 경우 task 자체가 좀 더 어려울 수는 있어도 fomulation은 상당히 간단하다.

#### 도식화된 표현
![25](https://user-images.githubusercontent.com/53552847/136954346-e8785bc4-3e09-4144-9134-0613d0f14975.PNG)
- 정답을 출력할 때, 모델의 출력값을 선형 layer에 넣어서 sequence length 내의 각 위치마다 들어가야할 단어를 선택하는 방식으로 예측을 진행한다.
- 정해진 횟수 또는 전체 길이의 수만큼 반복하여 단어를 하나씩 생성하여 이를 붙여 최종 답안으로 예측한다.
- 아주 일반적인 decoding 방법론이다.

### 2-4. Fine Tuning
#### BART
![26](https://user-images.githubusercontent.com/53552847/136954348-81a14467-ef1b-41a2-a0a4-47264a767675.PNG)
- 기계 독해, 기계 번역, 요약, 대화 등 seq2seq 문제에서 pre-training을 위한 denoising autoencoder라고 부른다.
- BERT는 Encoder만 존재하기 때문에 encoder output feature를 각 토큰마다 수치화시켜 시작과 끝을 예측하는 방식으로 진행된다.
- BART의 경우, 실제 text를 내보내는 방식으로 진행된다.
- pre-training을 진행할 때 역시 BERT의 경우 기존 문장에 몇몇 단어들을 MASK한 후 이를 예측하는 방식으로 학습하지만 BART의 경우 기존 문장을 비슷한 방식으로 MASK를 진행하지만 MASK된 단어를 맞추는 방식이 아닌 정답을 생성하는 방식으로 진행한다.

#### BART Encoder & Decoder
![27](https://user-images.githubusercontent.com/53552847/136954350-0b78a363-1a40-49be-9a05-0cebc8ad5071.PNG)
- BART의 인코더는 BERT처럼 Bi-directional
- BART의 디코더는 GPT처럼 uni-directional(autoregressive)

#### Pre-training BART
![28](https://user-images.githubusercontent.com/53552847/136954353-53d4e9b4-6b53-4f54-883a-5502d38601bd.PNG)
- 텍스트에 노이즈를 주고 원래 텍스트를 복구하는 문제를 푸는 방식으로 pre-training을 진행한다.

### 2-5. Post-preprocessing
- BART의 경우 decoding하는 방식으로 text를 생성하기 때문에, 이전 step에서 나온 출력을 다음 step의 입력으로 들어가는 방식을 채택한다.
- Greedy Seraching
    - 가장 높은 확률을 가지는 방향으로 단어를 생성해 나가는 방법
    - 생성은 빠르지만, 실제로 처음 선택한 결저이 나중에 안좋은 선택으로 이어질 수 있다.
- Exhaustive Search
    - 모든 단어 쌍들을 보는 방법
    - 가능한 가지수가 time step에 비례하여 exponential로 증가하여 문장의 길이가 조금만 길어져도 불가능하다.
    - 또한, vocabulary가 조금만 커져도 불가능하다.
- Beam Search
    - 일반적으로 모든 모델에서 채택되는 방법론이다.
    - exhaustive search를 진행하되, 각 time step마다 가장 높은 top-k만 채택하는 방식이다.

## 3. Extraction-based MRC vs Generation-based MRC
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
||Extraction-based MRC|Generation-based MRC|
|접근|지문 내에 존재하는 정답의 Start point/End point를 파악|질문에 대한 정답 생성|
|평가 방법|EM(Exact Match), F1 score|ROUGE-L, BLEU (Extraction-based MRC와 같은 평가 방법을 사용할 수 있지만 일반적인 생성 문제에 비슷하게 접근하기 위해 ROUGE-L, BLEU를 사용하기도 한다.)|
|정답 생성|모델 output을 Score로 바꾸고 이를 Start/End point로 변환하는 작업 추가|모델에서 바로 정답 추출|
|모델 구조|Encoder + Classifier|Seq2seq(Encoder-Decoder)|
|Prediction|지문 내 답이 위치|Free-form text|
|Loss|위치에 대한 확률 분포를 Negative Log Likelihood로 계산하여 접근|실제 text를 decoding할 때, teacher forcing과 같은 방식으로 학습|
|Special Token|[CLS], [SEP], [PAD] 등|정해진 텍스트 포맷으로 생성하여 사용한다, (e.g. question, context)|
|token_type_ids|BERT를 주로 사용하며, token_type_ids 존재|BART를 주로 사용하며, token_type_ids가 존재하지 않는다.|
|post-preprocessing|score 순으로 top-k -> 불가능한 조합 삭제 -> score가 가장 높은 조합 예측|Beam Search|
{: .tablelines}
