---
layout: post
title: "Second P-stage 3(Day 39)"
categories: boostcamp
tags: main
comments: true
---
BERT 기반 단일 문장 분류 모델 학습 방법에 대하여 학습한다.

**부스트 캠프 39일차 학습 요약**
- **학습** : BERT 기반 단일 문장 분류 모델 학습, BERT 기반 두 문장 관계 분류 모델 학습, BERT 언어 모델 기반 문장 토큰 분류
- **P-Stage** : 강의 실습 진행
- **피어 세션**

## 목차
- [1. KLUE 데이터셋 소개](#1-klue-데이터셋-소개)
- [2. 단일 문장 분류 with BERT](#2-단일-문장-분류-with-bert)
- [3. 두 문장 관계 분류 with BERT](#3-두-문장-관계-분류-with-bert)
- [4. 문장 토큰 분류 with BERT](#4-문장-토큰-분류-with-bert)

## 1. KLUE 데이터셋 소개
- 한국어 자연어 이해 벤치마크 (Korean Language Understanding Evaluation, KLUE)
- 우리가 살면서 직면하게 되는 모든 자연어 task 유형들을 모두 가지고 있으므로, KLUE에서 소개된 자연어 task를 모두 해결한다면 사실상 모든 자연어 문제를 전부 해결할 수 있다고 이해할 수 있다.

### 1-1. KLUE 종류 및 내용
- 문장 분류, 관계 추출
    - 단일 문장 분류 task
- 문장 유사도
    - 문장 임베딩 벡터의 유사도 (CLS token을 사용하여 두 문장의 cosine similarity 파악)
- 자연어 추론
    - 두 문장 관계 분류
- 개체명 인식, 품사 태깅, 질의 응답
    - 문장 토큰 분류
- 목적형 대화
    - DST 강의 참고
- 의존 구문 분석
    - 단어들 사이의 관계를 분석하는 task
    - 특징
        - 지배소 : 의미의 중심이 되는 요소
        - 의존소 : 지배소가 갖는 의미를 보완해주는 요소 (수식)
        - 어순과 생략이 자유로운 한국어와 같은 언어에서 주로 연구된다.
    - 분류 규칙
        - 지배소는 후위 언어이다. 즉, 지배소는 항상 의존소보다 뒤에 위치한다.
        - 각 의존소의 지배소는 하나이다.
        - 교차 의존 구조는 없다.
    - 분류 방법
        - Sequence Labeling 방식으로 처리 단계를 나눈다.
        - 앞 어절의 의존소가 없고 다음 어절이 지배소인 어절을 삭제하며 의존 관계를 만든다.
![16](https://user-images.githubusercontent.com/53552847/135734835-685be53f-1811-48ba-a3d7-337d7ae3c503.PNG)
        - 의존소를 찾는다.
        - 의존소에 매칭되는 지배소가 있을 경우, 하나씩 삭제해 나가면서 매칭한다.
    - 어디에 사용하는 가?!
        - 복잡한 자연어 형태를 그래프로 구조화해서 표현 가능하다.
        - 각 대상에 대한 정보 추출이 가능하다.
        - 예를 들어, "내가 그린 구름그림은 새털구름을 그린 구름그림이다."라는 문장이 있다고 했을 때, 다음의 과정으로 진행된다.
            - 구름그림 -> 새털구름 그림
            - 새털구름을 그린 구름 그림 -> 내가 그린 그림
            - 즉, 나는 구름 그림을 그렸다. -> 그 구름 그림은 새털구름 그림이다. 라는 정보를 획득할 수 있다.

## 2. 단일 문장 분류 with BERT
### 2-1. 단일 문장 분류 task
- 주어진 문장이 어떤 종류의 범주에 속하는 지를 구분하는 task
- 감성 분석
    - 긍부정, 중립 등 성향을 분류하는 task
    - 모니터링, 고객지원 또는 댓글 필터링 등을 자동화하는 작업에 주로 사용
    - 활용 방안
        - 혐오 발언 분류 : 혐오 발언 분류하여 조치를 취한다.
        - 기업 모니터링 : 기업 이미지, 브랜드 선호도, 제품평가 등을 분석한다.
- 주제 라벨링
    - 문장의 내용을 이해하고 적절한 범주를 분류하는 프로세스
    - 주제별로 뉴스 기사를 구성하는 등 데이터 구조화와 구성에 용이
    - 활용 방안
        - 대용량 문서 분류 : 대용량의 문서를 범주화
        - VoC(Voice of Customer) : 고객의 피드백을 제품 가격, 개선점, 디자인 등 적절한 주제로 분류하여 데이터를 구조화
- 언어 감지
    - 문장이 어떤 나라 언어인지를 분류하는 프로세스
    - 주로 번역기에서 정확한 번역을 위해 입력 문장이 어떤 나라의 언어인지 타겟팅하는 작업 가능
    - 활용 방안
        - 번역기 : 번역할 문장에 대해 적절한 언어 감지
        - 데이터 필터링 : 타겟 언어 이외 데이터 필터링
- 의도 분류
    - 문장이 가진 의도를 분류하는 프로세스
    - 입력 문장이 질문, 불만, 명령 등 다양한 의도를 가질 수 있기 떄문에 적절한 피드백을 줄 수 있는 곳으로 라우팅 작업 가능
    - 활용 방안
        - 챗봇 : 문장의 의도인 질문, 명령, 거절 등을 분석하고 적절한 답변을 주기 위해 활용 

### 2-3. 문장 분류를 위한 데이터
- Kor_hate
![17](https://user-images.githubusercontent.com/53552847/135734837-346d7476-05e7-4167-94fb-dd1ce3c250e6.PNG)
    - 혐오 표현에 대한 데이터
    - 특정 개인 또는 집단에 대한 공격적 문장
    - 무례, 공격적이거나 비꼬는 문장
    - 부정적이지 않은 문장
- Kor_sarcasm 
![18](https://user-images.githubusercontent.com/53552847/135734838-05d78a18-a3d6-4350-ba0d-1af796ae5d5b.PNG)
    - 비꼬지 않은 표현의 문장
    - 비꼬는 표현의 문장
- Kor_sae
![19](https://user-images.githubusercontent.com/53552847/135734839-19c43f1c-73c6-4ab6-abb9-4f91a101dc1d.PNG)
    - 예/아니오로 답변 가능만 질문
    - 대안 선택을 묻는 질문
    - Wh-질문 (who, what, where, when, why, how)
    - 금지 명령
    - 요구 명령
    - 강한 요구 명령
- Kor_3i4k 
![20](https://user-images.githubusercontent.com/53552847/135734844-fcd1ca29-93b2-4458-800a-d892ad2b8acd.PNG)
    - 단어 또는 문장 조각
    - 평서문
    - 질문
    - 명령문
    - 수사적 질문
    - 수사적 명령문
    - 억양에 의존하는 의도

### 2-4. 단일 문장 분류 모델 학습
- 모델 구조도
![21](https://user-images.githubusercontent.com/53552847/135734845-f7d45451-a04a-48bf-bd0c-5855c76df547.PNG)
    - BERT의 [CLS] token의 vector를 classification하는 Dense Layer 사용
- 주요 매개 변수
    - input_ids : Sequence Token을 입력
    - attention_mask : [0, 1]로 구성된 마스크, 패딩 토큰을 구분
    - token_type_ids : [0, 1]로 구성되며, 문장 구분
    - position_ids : 각 입력 시퀀스의 인덱스 임베딩
    - inputs_embeds : input_ids 대신 직접 임베딩 표현을 할당
    - lables : loss 계산을 위한 Label
    - next_sentence_label : 다음 문장 예측 loss 계산을 위한 Label
- 학습 과정
    - Dataset 다운로드 (HuggingFace Datasets)
    - Dataset 전처리 및 토큰화 (Pandas, HuggingFace Transformers Tokenizer)
    - DataLoader 설계 (PyTorch Dataset)
    - Train, Test Dataset 준비 (PyTorch Dataset)
    - TrainingArguments 설정 (HuggingFace Transformers TrainingArguments)
    - Pre-trained Model import (HuggingFace Transformers BERT (or etc))
    - Trainer 설정 (HuggingFace Transformers Trainer)
    - Model 학습 (HuggingFace Transformers Trainer)
    - Predict 함수 구현 및 평가



## 3. 두 문장 관계 분류 with BERT
- 두 문장 관계 분류 모델 학습은 주어진 두 문장에 대하여 두 문장에 대한 Label을 예측하는 것이다.
- 단일 문장 분류 모델과의 가장 큰 차이는 input 문장의 개수가 2개라는 점이다.
- 두 문장 관계 분류 모델에서는 2개의 문장을 입력으로 받아 그 문장들 사이의 관계에 해당하는 라벨을 예측하는 문제이다.

### 3-1. 두 문장 관계 분류 task
![22](https://user-images.githubusercontent.com/53552847/135734847-d4b83a30-3741-4939-afef-3307882f8f96.PNG)
- 주어진 2개의 문장에 대하여, 두 문장의 자연어 추론과 의미론적 유사성을 측정하는 task

### 3-2. 두 문장 관계 분류를 위한 데이터
- Natural Language Inference (NLI)
![23](https://user-images.githubusercontent.com/53552847/135734848-7464cdc9-c514-4578-81b9-c704a433e854.PNG)
    - 언어 모델이 자연어의 맥락을 이해할 수 있는지 검증하는 task
    - 전제문장과 가설문장을 entailment, contradiction, Neutral으로 분류한다.
- Semantic text pair
![24](https://user-images.githubusercontent.com/53552847/135734849-9bf9b7fd-b241-4396-97da-b6809e0f2c0a.PNG)
    - 두 문장의 의미가 서로 같은 문장인지 검증하는 task

### 3-3. 두 문장 관계 분류 모델 학습
- IRQA (Information Retrieval Question and Answering)
![25](https://user-images.githubusercontent.com/53552847/135734851-75510ee5-d6a1-4732-8e0b-7041d4c640c3.PNG)
    - 사전에 정의해 놓은 QA set에서 가장 적절한 답변을 찾아내는 과정
    - 순서
        - 질문 입력
        - BERT를 통한 sentence embedding
        - 사전에 정의한 table에서 Q, A가 pair로 되어있는 table 역시 sentence embedding을 한다.
        - 기존에 존재하던 Question과 나의 질문 Qeury 사이의 유사도를 비교해서 가장 적절한 문장을 반환한다. (여기까지는 기존의 cosine similarity를 이용한 chatbot을 만드는 과정과 동일하다.)
        - 이러한 task 뒷단에 Paraphrase Detection을 두 문장 관계 분류 task로 학습한 모델을 부착한다.
        - 기존의 과정에서 top N개의 문장을 출력할 수 있는데, 이 때 top 1이 정답 문장이 되지 않을 수도 있으므로, Paraphrase Detection을 부착함으로서, 내가 질의한 Query와 사전의 정의된 Question이 실제로 유사한 의미를 가지는 지 검증할 수 있다.
        - 이렇게 검증 filtering을 통과한 후에 최종 질문을 출력하고 이에 대응하는 답변을 보여준다.

## 4. 문장 토큰 분류 with BERT
- 문장 토큰 분류 모델은 전체 문장에 대한 하나의 Label이 아닌, 각 토큰에 대한 Label을 예측한다.
- 활용 분야로는 POS tagging, NER 등이 있다.

### 4-1. 문장 토큰 분류 task 소개
![26](https://user-images.githubusercontent.com/53552847/135734852-f75ed253-9797-4d01-bd48-b5bcdf3400b1.PNG)
- 주어진 문장의 각 token이 어떤 범주에 속하는지 분류하는 task
- NER (Named Entity Recognition)
    - 개체명 인식은 문맥을 파악해서 인명, 기관명, 지명 등과 같은 문장 또는 문서에서 특정한 의미를 가지고 있는 단어 또는 어구(개체) 등을 인식하는 과정을 의미한다.
    - 개체명 인식에서는 모델이 문맥을 정확히 파악하는가, 못하는가가 가장 중요하다. 같은 단어라도 다양한 개체로 인식될 수 있기 때문이다.
    - pororo 라이브러리
- POST TAGGING (Part-of-speech tagging)
    - 품사란 단어를 문법적 성질의 공통성에 따라 언어학자들이 몇 갈래로 묶어 놓은 것
    - 품사 태깅은 주어진 문장의 각 성분에 대하여 가장 알맞은 품사를 태깅하는 것
    - pororo 라이브러리

### 4-2. 문장 token 분류를 위한 데이터
- kor_ner
    - 한국해양대학교 자연어 처리 연구실에서 공개한 한국어 NER 데이터 셋
    - 일반적으로, NER 데이터셋은 pos tagging도 함께 존재한다.
    - Entity tag에서 B의 의미는 개체명의 시작(Begin)을 의미하고, I는 내부(Inside), O는 다루지 않는 개체명(Outside)를 의미한다.
    - 즉, B-PER은 인물명 개체명의 시작을 의미하며, I-PER은 인물명 개체명의 내부 부분을 뜻한다.
    - kor_ner 데이터셋에서 다루는 개체명은 다음과 같다.
![27](https://user-images.githubusercontent.com/53552847/135734853-12781715-eb70-4e8a-905f-7246457558c2.PNG)

### 4-3. 문장 토큰 분류 모델 학습
- NER fine-tuning with BERT
![28](https://user-images.githubusercontent.com/53552847/135734854-4b040250-7448-4ac1-90f0-41e7976cef41.PNG)
- 주의점
    - 정답은 아니지만, 형태소 단위의 토큰을 음절 단위 토크으로 분해하고, Entity tag 역시 음절 단위로 매핑시켜준다.
    - 이는 tokenizer의 문제일 수 있는데, wordpiece를 사요하면 개체를 잘못 인식할 수 있다.
