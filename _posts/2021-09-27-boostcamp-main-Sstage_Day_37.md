---
layout: post
title: "Second P-stage 1(Day 37)"
categories: boostcamp
tags: main
comments: true
---
인공지능과 자연어 처리에 대하여 이해하고, 자연어처리에서의 전처리에 대하여 학습한다.

**부스트 캠프 37일차 학습 요약**
- **행사** : 타운홀 미팅, 오피스아워
- **학습** : 인공지능과 자연어처리, 자연어의 전처리
- **P-Stage** : 강의 실습 진행
- **피어 세션**

## 목차
- [1. 인공지능과 자연어처리](#1-인공지능과-자연어처리)
- [2. 언어 모델](#2-언어-모델)
- [3. 자연어의 전처리](#3-자연어의 전처리)
- [4. 해야할 일](#4-해야할-일)

## 1. 인공지능과 자연어처리
### 1-1. 자연어 처리의 응용 분야
![1](https://user-images.githubusercontent.com/53552847/135622560-57dd3cb3-bd5c-4ccf-8463-0ad635d6d1af.PNG)
- 자연어를 컴퓨터가 이해할 수 있도록 수학적으로 이쁘게 인코딩이 된다면 디코딩을 통해 무엇이든 할 수 있다.
- 대부분의 자연어 처리는 '분류' 문제이고, '분류' 문제를 해결하기 위해 자연어를 벡터화(인코딩)하여 처리한다.

### 응용 분야
- 문서 분류
- 문법, 오타 교정
- 정보 추출
- 음성 인식 결과 보정
- 음성 합성 텍스트 보정
- 정보 검색
- 요약문 생성
- 기계 번역
- 질의 응답
- 기계 독해
- 챗봇
- 형태소 분석
- 개체명 분석
- 감성 분석
- 관계 추출
- 의도 파악
- 이미지 캡셔닝
- 텍스트 압축
- Paraphrasing
- 주요 키워드 추출
- 빈칸 맞추기
- 발음기호 변환
- 소설 생성
- 텍스트 기반 게임
- 오픈 도메인 QA
- 가설 검증

### 1-2. 자연어 단어 임베딩
#### 특징 추출과 분류
- 과정
    - '분류'를 위해서는 데이터를 수학적으로 표현할 수 있어야 한다.
    - 먼저, 분류 대상의 특징(Feature)를 파악 (Feature Extraction)
    - 파악한 특징을 바탕으로 분류 대상을 그래프 위에 표현 가능
    - 분류 대상들의 경계를 수학적으로 나눌 수 있음(Classification)
    - 새로운 데이터 역시 특징을 기준으로 그래프에 표현함으로서 어떤 그룹과 유사한지 파악할 수 있다.
![2](https://user-images.githubusercontent.com/53552847/135622563-8b5f083c-ae71-4889-bb8b-188b0ccce18b.PNG)
- 과거에는 사람이 직접 특징(Feature)를 파악해서 분류했다.
- 실제 복잡한 문제들에선 분류 대상의 특징을 사람이 파악하기 어려울 수 있다.
- 이러한 특징을 컴퓨터가 스스로 찾고 (Feature Extraction), 스스로 분류(Classification) 하는 것이 '기계 학습'의 핵심이다.

### 1-3. Word2Vec
- "어떻게 하면 자연어를 좌표평면 위에 표현할 수 있을까?"라는 질문으로부터 파생되었다.
- 가장 단순한 표현 방식은 one-hot encoding 방식이며 이는 Sparse Representation이라고 한다.
![3](https://user-images.githubusercontent.com/53552847/135622565-9c210e81-3829-44a2-9d92-f26ea9de687d.PNG)
- 이러한 방식으로 진행하게 될 경우, n개의 단어가 주어졌을 경우 n차원 벡터로 표현하므로 굉장히 Sparse하게 되므로, 단어의 의미를 벡터 공간을 보고 파악하는 것이 불가능해진다.
- 이러한 문제로 인하여, one-hot encoding을 보완하기 위해 Word2Vec이 등장하였다.

#### Word2Vec 알고리즘
- 자연어(특히, 단어)의 의미를 벡터 공간에 임베딩.
- 한 단어의 주변 단어들을 통해 그 단어의 의미를 파악.
- 주변부의 단어를 예측하는 방식으로 학습 (Skip-Gram 방식)
- 단어에 대한 Dense Vector를 얻을 수 있다.
- 단어가 가지는 의미 자체를 다차원 공간에 '벡터화'하는 것
- 중심 단어의 주변 단어들을 이용해 중심단어를 추론하는 방식으로 학습된다.
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|장점|단점|
|단어간의 유사도 측정에 용이|단어의 Subword Information 무시|
|단어간의 관계 파악에 용이|Out-Of-Vocabulary에서 적용이 불가능|
|벡터 연산을 통한 추론 가능|.|
{: .tablelines}

### 1-4. FastText
- Word2Vec의 단점을 보완하기 위해 등장하였다.
- 한국어의 경우 다양한 용언의 형태를 가지는데, Word2Vec의 경우, 다양한 용언 표현들이 서로 독립된 vocab으로 관리된다.
- FastText는 subword information에 집중해서 만들어진 Word2Vec 알고리즘이다.
- 단어를 n-gram으로 분리한 후, 모든 n-gram vector를 합산한 후 평균을 통해 단어 벡터를 획득한다.
![4](https://user-images.githubusercontent.com/53552847/135622567-3739fa64-02bf-4b87-bd9f-a588b4837971.PNG)
- Word2Vec의 경우, orange와 oranges가 전혀 다른 단어로서 vector로 받아들이기 때문에, 만약 oranges가 vocab에 없다면, oov(out-of-vocabulary)로 받아들여 사용할 수 없다.
- FastText의 경우, orange, oranges를 n-gram으로 쪼개게 되어, 마지막에 's'가 붙는 경우를 제외하고는 대부분이 유사함을 알 수 있기 때문에, oranges와 orange의 임베딩 벡터를 구했을 때, 굉장히 유사하게 얻을 수 있게된다.
- 특히, FastText는 오탈자, OOV, 등장 횟수가 적은 학습 단어에 대하여 강세를 가진다.

### 1-5. Word Embedding의 한계
- Word2Vec이나 FastText와 같은 Word Embedding 방식은 동형어, 다의어 등에 대해서는 Embedding 성능이 좋지 못하다는 단점이 있다.
- 주변 단어를 통해 학습이 이루어지기 때문에, 문맥을 고려할 수 없다.
- 주변 문맥을 이용할 수 있는 언어 Embedding 알고리즘이 필요하다.
- 이로 인해, 딥러닝 기반의 자연어 처리 모델인 Language Model이 발전했다.

## 2. 언어 모델
### 2-1. 언어 모델
- 자연어의 법칙을 컴퓨터로 모사한 모델
- 주어진 단어들로부터 그 다음 등장할 단어의 확률을 예측하는 방식으로 학습 (이전 state로 미래 state를 예측)
- 다음의 등장할 단어를 잘 예측하는 모델은 해당 언어의 특성이 잘 반영된 모델이자, 문맥을 잘 계산하는 좋은 언어 모델이다.

### 2-2. Markov 기반의 언어 모델
- Markov Chain Model
- 초기의 언어 모델은 다음 단어나 문장이 나올 확률을 통계와 단어의 n-gram 기반으로 계산
- 딥러닝 기반의 언어 모델은 해당 확률을 최대로 하도록 네트워크를 학습한다.
![5](https://user-images.githubusercontent.com/53552847/135622572-3b4f9e93-02b8-44ed-8b38-a34e6723d86c.PNG)

### 2-3. Recurrent Neural Network 기반의 언어모델
![6](https://user-images.githubusercontent.com/53552847/135622578-d99331f4-ec40-4ecf-9ea2-55ce3649b43b.PNG)
- Markov 기반의 언어 모델을 그대로 DeepLearing Network로 옮긴 예제가 RNN 기반 언어 모델이다.
- RNN은 히든 노드가 방향을 가진 엣지로 연결되 순환 구졸르 이룬다.(Directed Cycle)
- 이전 state 정보가 다음 state를 예측하는데 사용됨으로서, 시계열 데이터 처리에 특화되어 있다.
- 마지막 출력은 앞선 단어들의 '문맥'을 고려해서 만들어진 최종 출력 vector이며 이를 Context Vector라고 부른다.
- 출력된 Context Vector 값에 대하여 Classification Layer를 붙이면 문장 분류를 위한 신경망 모델이 완성된다.

### 2-3. RNN 기반 Seq2Seq
![7](https://user-images.githubusercontent.com/53552847/135622579-bea23cb3-7b21-4b6d-9edd-f38e8fcc0f0b.PNG)
- Encoder Layer : RNN 구조를 통해 Context Vector를 획득한다.
- Decoder Layer : 획득된 Context Vector를 입력으로 출력을 예측한다.
- Seq2Seq, 즉 Encoder/Decoder 방식으로 접근하게 되면, 모든 자연어 처리 어플리케이션을 개발할 수 있다.
- RNN 구조의 문제점
    - 입력 Sequence의 길이가 길어질수록, 처음 token에 대한 정보가 희석된다.
    - 고정된 Context Vector 사이즈로 인하여 긴 Sequence에 대한 정보를 함축하기 힘들다.
    - 모든 Token이 영향을 미치기 때문에, 중요하지 않은 Token 역시 영향을 미친다.
- 이러한 RNN의 문제를 해결하기 위하여 Attention 모델이 등장하였다.

### 2-4. Attention 모델
- 인간이 정보를 처리할 때, 모든 Sequence를 고려하면서 정보 처리를 하는 것은 아니다.
- 인간의 정보처리와 마찬가지로, 중요한 Feature는 더욱 중요하게 고려하는 것이 Attention의 모티브이다.
- RNN에서는 Encoder의 마지막에 출력되는 Context Vector에만 관심을 가졌던 것에 반해, Attention에서는 각각의 RNN cell에서의 hidden state vector를 사용한다.
- 문맥에 따라 동적으로 할당되는 encode의 Attention Weight로 인하여 Dynamic context vector를 획득할 수 있다.
- Seq2Seq의 encoder, decoder의 성능을 비약적으로 향상시킨다.
![8](https://user-images.githubusercontent.com/53552847/135622580-200f3987-bafe-48dd-aca9-ee685f25648d.PNG)
- Attention Model의 한계 : 여전히 RNN이 순차적으로 연산이 이뤄짐에 따라 연산 속도가 느리다.

### 2-5. Self-Attention 모델
![9](https://user-images.githubusercontent.com/53552847/135622583-511bbf8a-a53f-4f5d-ad5d-e2bea790eb05.PNG)
- "이전 RNN 방식대로 순차적으로 하지 않으면서 모든 token들을 한 번에 연결할 수 있지 않을까?"라는 질문으로 부터 파생되었따.
- Transformer(Self-Attention)은 하나의 Network 내에 Encoder, Decoder를 합쳐진 구조를 가지고 있고 다음과 같다고 이해할 수 있다.
![10](https://user-images.githubusercontent.com/53552847/135622586-1168a526-a13c-4147-8503-1446080ad40b.PNG)

## 3. 자연어의 전처리
### 3-1. 자연어 전처리
#### 전처리
- 원시 데이터를 기계 학습 모델이 학습하는데 적합하게 만드는 프로세스
- 학습에 사용될 데이터를 수집&가공하는 모든 프로세스

#### 자연어 전치리. 왜 필요하지?!
- Task의 성능을 가장 확실하게 올릴 수 있는 방법이다.
- 모델을 아무리 바꾸고, 튜닝하더라도, 데이터 자체에 쿤제가 있다면 성능이 나올 수 없다.
- 가장 중요한 것은 데이터이다.

#### 자연어 처리 단계
- task에 따라 순서는 달라질 수 있지만 보통 아래의 표와 같은 순서를 가진다.
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|자연어 처리 단계|예시|
|Task 설계|악성 댓글 Classifier 만들기|
|필요 데이터 수집|댓글 데이터 수집|
|통계학적 분석|Token 개수를 바탕으로 Outlier 제거, 빈도 수를 바탕으로 Vocab 정의|
|전처리|개행문자 제거, 특수문자 제거, 문장분리 보정 등|
|Tagging(Labeling)|악성 댓글인지 아닌지 Label을 매긴다.|
|Tokenizing|자연어를 어떤 단위로 살펴볼 것인지 결정한다.(어절, 형태소, 음절, wordpiece 등)|
|모델 설계|.|
|모델 구현|.|
|성능 평가|.|
{: .tablelines}
- 위의 표의 전처리에는 다음과 같은 것들을 할 수 있다.
    - 개행문자 제거
    - 특수문자 제거
    - 공백 제거
    - 중복표현 제거
    - 이메일, 링크 제거
    - 제목 제거
    - 불용어 제거
    - 조사 제거
    - 띄어쓰기 및 문장분리 보정

### 3-2. 자연어 전처리를 위한 Python String 관련 함수
- 전처리를 하기 위해서는 Python String 함수와 친해져야 한다.
- 대소문자 변환
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|함수|설명|
|upper()|모두 대문자로 변환|
|lower()|모두 소문자로 변환|
|capitalize()|문자열의 첫 문자를 대문자로 변환|
|title()|문자열에서 각 단어의 첫 문자를 대문자로 변환|
|swapcase()|대문자, 소문자를 서로 변환|
{: .tablelines}

- 편집, 치환
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|함수|설명|
|strip()|좌우 공백 제거|
|rstrip()|오른쪽 공백 제거|
|lstrip()|왼쪽 공백 제거|
|replace(a,b)|a를 b로 치환|
{: .tablelines}

- 분리, 결합
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|함수|설명|
|split()|공백으로 분리|
|split('\t')|탭을 기준으로 분리|
|''.join()|주어진 리스트에 대하여 각 요소사이의 공백을 두고 결합|
|lines.splitliness()|라인 단위로 분리|
{: .tablelines}

- 구성 문자열 판별
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|함수|설명|
|isdigit()|숫자 여부 판별|
|isalpha()|영어, 알파벳 여부 판별|
|isalnum()|숫자 혹은 영어, 알파벳 여부 판별|
|islower()|소문자 여부 판별|
|isupper()|대문자 여부 판별|
|isspace()|공백 문자 여부 판별|
|startswith()|문자열이 주어진 문자로 시작하는지 여부 파악|
|endswith()|문자열이 주어진 문자로 끝나는지 여부 파악|
{: .tablelines}

- 검색
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|함수|설명|
|count(str)|문자열에서 주어진 문자가 출현한 빈도 리턴|
|find(str)|문자열에서 주어진 문자가 처음으로 등장한 위치 리턴. 존재하지 않을 경우 -1 리턴|
|find(str, index)|문자열에서 주어진 index부터 검사하여 주어진 문자가 처음으로 등장한 위치 리턴. 존재하지 않을 경우 -1 리턴|
|rfind(str|문자열에서 오른쪽부터 검사하여 주어진 문자가 처음으로 등장한 위치 리턴. 존재하지 않을 경우 -1 리턴|
|index()|find와 비슷하지만 존재하지 않는 경우에 예외 발생|
|rindex|rfind와 비슷하지만 존재하지 않는 경우에 예외 발생|
{: .tablelines}

### 3-3. 한국어 토큰화
- 토큰화 (Tokenizing)
    - 주어진 데이터를 토큰(Token)이라고 불리는 단위로 나누는 작업
    - 토큰이 되는 기준은 다를 수 있다.(음절, 어절, 단어, 형태소, 자소, wordpiece 등)
- 문장 토큰화 (Sentence Tokenizing) 
    - 문장 분리
- 단어 토큰화 (Word Tokenizing)
    - 구두점 분리, 단어 분리
- 영어는 New York과 같은 합성어 처리와 it's와 같은 줄임말 예외처리만 하면, 띄어쓰기를 기준으로도 잘 동작하는 편이다.
- 한국어는 조사나 어미를 붙여 만든 교착어로서 띄어쓰기만으로는 부족하다.
- 한국어에서는 어절의 의미를 가지는 최소 단위의 형태로소 분리하곤 한다.

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
