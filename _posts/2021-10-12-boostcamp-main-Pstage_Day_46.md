---
layout: post
title: "Third P-stage 1(Day 46)"
categories: boostcamp
tags: main
comments: true
---
MRC Project에 들어가기 앞서 MRC에 대한 Introduction을 진행한다.

**부스트 캠프 46일차 학습 요약**
- **학습** : MRC Intro & Python Basics
- **피어 세션**

## 목차
- [1. Introduction to MRC](#1-introduction-to-mrc)
- [2. Unicode & Tokenization](#2-unicode--tokenization)
- [3. Dataset](#3-dataset)
- [4. 해야할 일](#4-해야할-일)

## 1. Introduction to MRC
### 1-1. Machine Reading Comprehension (MRC)
- 주어진 지문(context)를 이해하고, 주어진 질의 (Query/Question)의 답변을 추론하는 문제
- 질문이 들어오면 Search Engine을 통해 지문을 찾고 지문 내에서 정답을 찾는 방식

### 1-2. MRC 종류
- Extractive Answer Dataset
    - 질의에 대한 답이 항상 주어진 지문의 segment or span으로 존재
    - 답변이 지문 내에 항상 존재한다고 가정한다.
    - Cloze Test
![1](https://user-images.githubusercontent.com/53552847/136954355-5efdf501-e5f2-4314-8387-1228d07b5cff.PNG)
        -  질문 문장에서 빠진 단어를 알아 맞추는 것을 목표로 한다.
        - (CNN/Daily Mail Dataset) 
    - Span Extraction
![2](https://user-images.githubusercontent.com/53552847/136954357-19dcd805-4806-47ef-ae58-72174b0f505c.PNG)
        - 질문에 대한 답을 지문에서 찾는 방식이다.
        - SQuAD, KorQuAD, NewsQA, Natural Questions가 있다.
    - Descriptive/Narrative Answer Dataset
![3](https://user-images.githubusercontent.com/53552847/136954360-2a1bab72-4330-4d0e-b1a1-a67472d48b47.PNG)
        - 답이 지문 내에서 추출한 span이 아니라 질의를 보고 생성된 sentence(or free-form) 형태이다. 
        - MS MARCO(Bajaj et al., 2016)이 있다.
    - Multiple-choice Dataset
        - 질의에 대한 답을 여러개의 answer candidates 중 하나를 고르는 형태이다.
        - MCTest, RACE, ARC, etc
        - QA 모델을 만드는 차원에서는 적합하지 않기 때문에 실제로 사용되지는 않는다.

### 1-3. MRC Dataset 역사
![4](https://user-images.githubusercontent.com/53552847/136954364-6a8b0985-d99e-4f17-b5d1-cf73dd54c9ab.PNG)
- MCTest를 시작점으로 보는 분들이 대다수 이다.
- 위의 이미지와 같은 역사를 바탕으로 현재까지 이어지고 있다.

### 1-4. Challenges in MRC
- 단어들의 구성이 유사하지 않지만 동일한 의미의 문장을 이해해야 하는 경우
    - DuoRC(Paraphrased Paragraph)와 QuoRef(Coreference resolution) dataset을 해결할 때 어려움이 있다.
    - Paraphrased의 경우 같은 의미의 문장이지만 다른 단어들로 구성될 경우를 의미한다. 기계 입장에서 단어가 달라지면 같은 의미라고 판단하기가 어려워진다.
    - Coreference resolution은 대명사(그, 그녀)가 누구를 지칭하고 있는지 명확히 알 수 없어 기계 입장에서 이해하기 어려울 수 있다. 이로부터 대명가 누구를 지칭하고 있는지를 찾아내는 task가 중요하고 이로부터 MRC를 조금 더 원활하게 진행할 수 있다.
- 지문 내에 답변이 존재하지 않을 경우
    - 실제로 지문에 답변이 존재하지 않을 수 있지만, 모델 입장에서는 답변이 존재하는 것처럼 인식하여 잘못된 답변을 출력할 수 있다.
- Multi-hop reasoning
![5](https://user-images.githubusercontent.com/53552847/136954365-5a02e6d8-2184-448a-a2cb-6390ca762a66.PNG)
    - 때에 따라서는 다양한 지문이 있어야지만 질문에 해당하는 답을 찾을 수 있다.  
    - 여러 개의 Document에서 질의에 대한 Supporting Fact를 찾아야지만 답을 찾을 수 있다.
    - 위의 그림처럼, 질문에 대한 답이 여러개의 Document를 거쳐 정보를 취합해야 얻을 수 있다.
    - EX) HotpotQA, QAngaroo

### 1-5. MRC 평가방법
![6](https://user-images.githubusercontent.com/53552847/136954366-11d4c300-50a1-4009-96c3-2ca30aa5823e.PNG)
- Exact Match(= Accuracy)
    - 예측한 답과 Ground-Truth가 정확히 일치하는 샘플의 비율
    - (Number of correct samples) / (Number of whole samples)
    - 답변이 조금만 달라져도 점수를 얻지 못하는 단점이 있다.
- F1 Score
    - 예측한 답과 Ground-Truth 사이의 Token Overlap을 F1으로 계산한다.
    - descriptive answer에 약점을 가진다. f1의 경우 단어의 overlap을 보다보니 답변 자체의 언어적인 부분에서의 변화를 보기 힘든부분이 존재한다.
- ROUGE-L
    - 예측한 값과 Ground-Truth 사이의 Overlap Recall
    - n-gram, 즉 여러개의 단어가 겹치는지 안겹치는지를 LCS(Longest Common Subsequence)를 기반으로 찾아서 점수를 매긴다.
- BLEU
    - 예측한 답과 Ground-Truth 사이의 Precision
    - 실제로 n-gram에서의 n을 정의해줌으로서, n-gram level에서의 겹치는 단어를 비교하여 점수를 낸다.

## 2. Unicode & Tokenization
### 2-1. Unicode란?
![7](https://user-images.githubusercontent.com/53552847/136954369-a66c9248-bcb4-42a9-bb68-d9a8477e8b71.PNG)
- 전 세계의 모든 문자를 일관되게 표현하고 다룰 수 있도록 만들어진 문자셋
- 각 문자마다 숫자 하나에 매핑한다.

### 2-2. 인코딩 & UTF-8
- 인코딩이란?
    - 문자를 컴퓨터에서 저장 및 처리할 수 있도록 이진수로 바꾸는 것
- UTF-8 (Unicode Transformation Format)
    - UTF-8은 현재 가장 많이 쓰는 인코딩 방식
    - 문자 타임에 따라 다른 길이의 바이트를 할당한다.
![8](https://user-images.githubusercontent.com/53552847/136954371-7a8d4ce7-6e58-4385-bf46-97230dbd58b4.PNG)
![9](https://user-images.githubusercontent.com/53552847/136954372-dbc16cb7-38bf-4efe-a3ef-26193764b4e7.PNG)

### 2-3. Python에서 Unicode 다루기
![10](https://user-images.githubusercontent.com/53552847/136954375-1aae9982-58ba-4ed9-b334-1f174ec9a876.PNG)
- Python3부터 string 타입은 유니코드 표준을 사용한다.
- `ord` : 문자를 유니코드 code point로 변환한다.
- `chr` : Code point를 문자로 변환한다.

### 2-4. Unicode와 한국어
- 한국어는 한자 다음으로 유니코드에서 많은 코드를 차지하고 있다.
- 완성형
    - 현대 한국어의 자모 조합으로 나타낼 수 있는 모든 완성형 한글 11,172자(가-힣)
    - U+AC00 ~ U+D7A3
- 조합형
    - 조합하여 글자를 만들 수 있는 초,중,종성
    - U+1100 ~ U+11FF, U+A960 ~ U+A97F, U+D7B0 ! U+D7FF
- 조합형의 경우, len을 활용하여 길이를 확인할 때 조합한 개수만큼의 길이가 나오기 때문에 완성형으로 만든 단어와 조합형으로 만든 단어의 길이를 출력할 때 다르게 나올 수 있다.

### 2-5. Tokenizing
- 텍스트를 토큰 단위로 나누는 것
- 단어(띄어쓰기 기준), 형태소, subword 등 여러 토큰 기준이 사용된다.
- Subword Tokenizing
![11](https://user-images.githubusercontent.com/53552847/136954377-b40695ab-c5f2-4ef2-a909-6d5f0a26a276.PNG)
    - 자주 쓰이는 글자 조합은 한 단위로 취급하고, 자주 쓰이지 않는 조합은 subword로 쪼갠다. (Byte Pair Encoding)
    - '##'은 디코딩(토크나이징의 반대 과정)을 할 때, 토큰을 앞 토큰에 띄어쓰기 없이 붙인다는 의미를 뜻한다.

### 2-6. BPE (Byte Pair Encoding) 
![12](https://user-images.githubusercontent.com/53552847/136954378-fa66fc88-98b2-4480-8eb7-218531a0e710.PNG)
- 데이터 압축용으로 제안된 알고리즘 이다.
- NLP에서 Tokenizing용으로 활발히 사용되고 있다.
    - 가장 자주 나오는 글자 단위 Bigram(or Byte Pair)를 다른 글자로 치환한다.
    - 치환된 글자를 저장해둔다.
    - 위 두 과정을 반복하여 진행한다.
- 이런 과정을 바탕으로 subword tokenizer가 진행되며, 위의 문장에서 아버지는 한 단어가되고 다른 단어들은 쪼개져서 나오는 것을 볼 수 있다. 

## 3. Dataset
### 3-1. KorQuAD
- LG CNS가 AI 인공지능 연구를 위해 공개한 질의응답/기계독해 한국어 데이터셋
- 한국어 위키피디아 1,550개의 문서에 대한 10649건의 하위 문서들과 크라우드 소싱을 통해 제작한 63,952개의 질의응답 쌍으로 구성 (Train : 60407, Dev : 5774, Test : 3898)
- 누구나 데이터를 내려받고, 학습한 모델을 제출하고 공개된 리더보드에 평가를 받을 수 있다. 이로인해 객관적인 기준을 가진 연구 결과 공유가 가능해진다.
- 현재 v1.0, v2.0이 공개되어 있으며, 2.0은 보다 긴 분량의 문서가 포함되어 있어 단순 자연어 문장 뿐만 아니라 복잡한 표와 리스트 등을 포함하는 HTML 형태로 표현되어 있어 문서 전체 구조에 대한 이해가 필요하다.
- 영어 dataset의 Natural Questions와 형태가 상당히 비슷하다.

### 3-2. KorQuAD 수집 과정
- SQuAD v1.0의 데이터 수집 방식을 벤치마크하여 표준성을 확보함.
- 대상 문서 수집
    - 위키 백과에서 수집한 글들을 문단 단위로 정제, 이미지/표/URL 제거
    - 짧은 문단, 수식이 포함된 문단 등 제거
- 질문/답변 생성
    - 크라우드소싱을 통해 질의응답 70,000+ 쌍 생성
    - 작업자가 양질의 질의응답 쌍을 생성하도록 상세한 가이드라인을 제시
- 2차 답변 태깅
    - 앞에서 생성한 질문에 대해 사람이 직접 답해보면서 Human Performance 측정
    - 앞서 질문/답변 생성 과정에 참여한 사람은 참여 불가

### 3-3. HuggingFace datasets 라이브러리
![13](https://user-images.githubusercontent.com/53552847/136954379-4cd770d8-e426-41f3-b537-8f252fff66d4.PNG)
- HuggingFace에서 만든 datasets는 자연어처리에 사용되는 대부분의 데이터셋과 평가 지표를 접근하고 공유할 수 있게 만든 라이브러리이다.
- Numpy, Pandas, PyTorch, Tensorflow2 호환
- 접근가능한 모든 데이터셋이 memory-mapped, cached 되어 있어 데이터를 로드하면서 생기는 메모리 공간 부족이나 전처리 과정 반복의 번거로움 등을 피할 수 있다.
- KorQuAD의 경우 squad_kor_v1, squad_kor_v2로 불러올 수 있다.
- train set만 가져오고 싶을 경우, 위의 그림처럼 split option을 줘서 가져올 수 있다.

### 3-4. KorQuAD Example
![14](https://user-images.githubusercontent.com/53552847/136954382-16f09244-ee04-4532-890c-8796b7175043.PNG)
- KorQuAD 역시 SQuAD와 비슷한 형태를 띄고 있기 때문에, 질문에 대한 답이 항상 지문 내에 존재한다.
- 질문에 대한 답이 지문 내에서 몇 번째 Character에서 시작하는지를 알려주는 요소가 'answer_start'이다.
- 이런 answer_start 요소가 중요한 이유는 답변이 한 번만 나오면 문제가 되지 않지만, 경우에 따라 답변이 지문 내에서 여러 개가 나올 수 있지만 유추할 수 있는 문장은 하나만 있을 수도 있기 때문이다.
- answer start가 주어진 경우를 strong supervision이라고 볼 수 있고, 주어지지 않을 경우를 distance supervision이라고 볼 수 있다.
![15](https://user-images.githubusercontent.com/53552847/136954386-70e6cb67-7496-4996-aab9-ef0204488cb7.PNG)
- 실제로 print를 하면 위의 그림과 같다.
- answer start와 text가 list의 형태로 되어있는데, 답변이 여러개가 존재하는 경우가 있을 수 있기 때문이며, 많은 경우에는 답변이 하나이고, 실제 train 과정에서는 답변이 하나인 경우만 학습한다.
- validation/test dataset에서는 text에서의 답변이 한 개가 아닌 다른 답변도 인정해주기 위해서 list에 속한 데이터가 2~3개가 될 수도 있다.

## 4. 해야할 일
- Byte Pair Encoding 심화학습
    - 내용 이해
    - 쓰임새




