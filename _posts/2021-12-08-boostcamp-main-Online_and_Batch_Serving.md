---
layout: post
title: "Online and Batch Serving"
categories: boostcamp
tags: main
comments: true
---
Model Serving에 대하여 학습한다.

## 목차
- [1. Serving](#1-serving)
- [2. Online Serving](#2-online-serving)
- [3. Batch Serving](#3-batch-serving)
- [4. Online vs Batch](#3-online-vs-batch)

## 1. Serving
- Production(Real World) 환경에 모델을 사용할 수 있도록 베포
- 머신러닝 모델을 개발하고, 현실 세계(앱, 웹)에서 사용할 수 있게 만드는 행위
- 서비스화라고도 표현할 수 있다.
- 머신러닝 모델을 회사 서비스의 기능 중 하나로 활용한
    - ex) 추천 시스텀의 추천 알고리즘
- Input이 제공되면 모델이 예측 값(Output)을 반환
- Online Serving / Batch Serving이 존재
- 그 외, 클라이언트(모바일 기기, IoT Device)에서 Edge Serving도 존재

### 1-1. Web Server
- HTTP를 통해 웹 브라우저를 요청하는 HTML 문서나 object를 전송해주는 서비스 프로그램
- Request를 받으면 요청한 내용을 Response하는 프로그램
- Brower와 Server 사이의 Request and Response
    - 크롬 -> www.youtube.come (HTTP Request)
    - www.youtube.come -> 크롬 (HTTP Response)
- 실 서비스에서의 Request and Response
    - 고객 -> 쇼핑몰 (회원 가입 요청)
    - 쇼핑몰 -> 고객 (회원 가입 완료 응답)
- 머신러닝 모델 서버
    - Input을 제공하면 이에 대하여 모델을 사용해 예측하고 예측된 값을 반환하는 서버를 의미한다.
    - 클라이언트 -> 모델 서버 (HTTP Request)
    - 모델 서버 -> 클라이언트 (HTTP Response)

### 1-2. API
- Application Programming Interface
- 운영체제나 프로그래밍 언어가 제공하는 기능을 제어할 수 있게 만든 Interface
- Interface는 기계와 인간의 소통 창구라고 이해할 수 있다.
- 서비스에 해당 기능을 외부에서 사용할 수 있도록 노출 시킨 것. 
    - 기상청 API, 지도 API 등
    - 라이브러리 함수 또한 API이다. ex) Pandas, Tensorflow, PyTorch

## 2. Online Serving
- Request가 올 때마다 실시간으로 예측
- 클라이언트(Application)에서 ML 모델 서버에 HTTP 요청을 하고, 머신러닝 모델 서버에서 예측한 후 예측 값을 Response한다.
- 단일 데이터를 받아 실시간으로 예측하는 문제에 주로 사용한다?! (확인하기)

### 2-1. 특징
- ML 모델 서버에 Request할 때, ML 모델 서버에서 데이터 전처리를 진행해야 할 수도 있다.
    - 전처리 서버와 ML 모델 서버를 나눌 수도 있다.
- 서비스의 서버에 ML 서버를 포함하는 경우도 있고, ML 서버를 별도로 운영하는 경우도 있다.
- 회사에서 개발 조직과 데이터 조직의 협업하는 방식에 따라 다르게 개발될 수 있다.

### 2-2. 구현 방식
- 직접 API 웹 서버 개발
    - Flask, FastAPI 등을 사용해 서버 구축 
- 클라우드 서비스 활용
    - AWS의 SageMaker, GCP의 Vertext AI 등을 활용 
    - 장점
        - 직접 구축해야 하는 MLOps의 다양한 부분 (API 서버 만들기)이 만들어진다.
        - 사용자 관점에서 PyTorch를 사용하듯 학습 코드만 제공하면 API 서버가 만들어진다.
    - 단점
        - 클라우드 서비스에 익숙해야 잘 활용할 수 있다.
        - 내부 실행 구조를 잘 알아야 문제 상황이 발생했을 때, 잘 해결할 수 있다.
        - 비용 문제 -> 직접 만드는 것 보다 더 많은 비용이 들 수 있다.
    - 회사에 소수 인원만 존재하고, 소수 인원이 많은 업무를 해야할 때 사용하면 좋다.
    - 어떤 방식으로 AI 제품을 만들었는지 확인할 수 있어서 사용해보는 것이 좋다.
- Serving Library 활용
    - Tensorflow Serving, Torch Serve, MLFlow, BentoML 등을 활용 
    - 서버에 대한 이해가 충분하지 않을 경우 어려울 수 있다.
    - 다양한 방식으로 개발할 수 있지만, 매번 추상화된 패턴을 가질 수 있다.
    - 추상화된 패턴을 잘 제공하는 오픈소스를 활용하는 방식으로 진행한다.
- 추천 방식 (클라우드 비용에 대하여 괜찮다고 할 경우)
    - 첫번째 -> 프로토타입 모델을 클라우드 서비스를 활용해 베포
    - 두번째 -> 직접 FastAPI 등을 활용해 서버 개발
    - 세번쨰 -> Serving 라이브러리를 활용해 개발

### 2-3. 고려 사항
- Serving을 할 때, Python 버전, 패키지 버전 등의 Dependency가 굉장히 중요하다.
- '재현 가능하지 않은' 코드는 Risk를 가지고 있는 코드이다.
- Virtualenv, Poetry, Docker를 함꼐 사용하면 좋다.

#### Latency 최소화
- Latency: 하나의 예측을 요청하고 받는데 까지 걸리는 시간
- Latency가 길다 == Loading 시간이 길다
- 입력 Input 데이터 기반으로 추가적인 정보가 DB에 있을 경우
    - DB로부터 데이터를 추출하기 위해 쿼리를 실행하고, 결과를 받는 시간이 소요될 수 있다.
- 모델 수행 연산
    -  RNN, LSTM 등의 모델은 회귀 분석보다 많은 연산을 요구하고, 더 오래 걸린다.
    -  모델 경량화 작업이 필요할 수 있고, 복잡한 모델보다 간단한 모델을 사용하는 경우도 존재한다.
-  후처리
    - 결과 값에 대한 보정이 필요한 경우
    - 유효하지 않은 값이 예측 될 경우 이를 후처리 해주어야 한다.
- 위 문제를 해결하기 위한 다양한 방법
    - 데이터 전처리 서버 분리 (혹은 Feature를 미리 가공 - Feature Store)
    - 모델 경량화
    - 병렬 처리 (Ray)
    - 예측 결과 캐싱

## 3. Batch Serving
- 주기적으로 학습을 하거나 예측을 하는 방법
- 한 번에 많은 예측을 수행한다.
- Batch Serving 관련 Library는 따로 존재하지 않으며 함수 단위를 주기적으로 실행한다.
- Airflow, Cron Job 등으로 Scheduling 작업을 진행한다. (Workflow Scheduler)
- 예를 들면, 다음과 같다.
    - 추천 시스템의 경우, 하루 전에 생성된 컨텐츠에 대한 추천 리스트를 예측
    - 1시간 마다 생성된 데이터를 바탕으로 1시간 뒤의 수요 예측
- 실시간이 필요없는 대부분의 방식에서 활용 가능하다.
- 장점
    - python notebook에서 작성한 코드를 함수화하고 주기적으로 실행하는 간단한 구조이다.
    - Online Serving보다 구현이 수월하고, 간단하다.
    - 한 번에 많은 데이터를 처리하므로 Latency 문제가 없다.
- 단점
    - 실시간으로 활용할 수 없다.
    - Cold Start -> 오늘 새로 생긴 컨텐츠를 추천할 수 없다.

## 4. Online vs Batch
<style>
.tablelines table, .tablelines td, .tablelines th{
    border: 1px solid black;
    }
</style>
|---|Online Serving|Batch Serving|
|input 관점|데이터가 하나씩 요청되는 경우|여러 데이터가 한 번에 처리되는 경우|
|Output 관점|API 형태로 바로 결과를 반환, 혹은 서버와 통신이 필요한 경우|주기적으로 예측을 해도 괜찮은 경우|
{: .tablelines}

- 처음부터 Online Serving (API 형태)로 만들어야 할 필요는 없다.
- 실시간 모델 결과가 어떻게 활용되는지 생각한 후에 진행할 필요성이 있으며, 예측한 결과가 바로 사용되지 않는다면 Batch Serving으로 진행해도 좋다.
- 먼저 Batch Serving으로 모델을 운영하며 점점 API 형태로 변환하는 방식이 좋다.
- Batch Serving의 경우, 결과를 DB에 저장하며, 주기적으로 쿼리해서 조회하는 방식으로 활용할 수 있다.
