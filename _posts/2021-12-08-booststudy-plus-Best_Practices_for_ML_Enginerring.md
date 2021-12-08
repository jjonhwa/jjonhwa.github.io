---
layout: post
title: "[Reading] Rules of Machine Learning: Best Practices for ML Engineering"
categories: booststudy
tags: plus
comments: true
---
Model Serving에 대하여 이야기한다.

본 내용은 [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml)을 바탕으로 작성했다.

## 목차
- [1. Overview](#1-overview)
- [2. Before Machine Learning](#2-before-machine-learning)
- [3. ML phase 1: Your First Pipeline](#3-ml-phase-1-your-first-pipeline)
- [4 .ML phase 2: Feature Engineering](#4-ml-phase-2-feature-engineering)
- [5. ML phase 3: Slowed Growh, Optimization Refinement, and Complex Models](#5-ml-phase-3-slowed-growh-optimization-refinement-and-complex-models)

## 1. Overview
- 훌륭한 제품을 만들기 위해서는 훌륭한 Machine Learning 전문가가 아닌 훌륭한 Machine Learning Engineer처럼 Machine Learning을 수행해야 한다.
- 직면할 대부분의 문제는 engineering 문제이다.
- 대부분의 이득은 ML algorithm이 아닌 좋은 feature로 부터 나온다.
- Basic Approach
    - End to End 형태의 견고한 Pipeline
    - 합리적인 Objective로 시작하기
    - 간단한 방법으로 common-sense feature 추가하기
    - Pipeline이 견고하도록 유지하기.
    - 더 멀리 갈 수 없을 때만 이러한 Approch에서 벗어나도록 하라. 복잡해질 수록, 이후 release가 느려진다.

## 2. Before Machine Learning
- 이 section은 기계 학습 시스템을 구축하기에 적절한 시기인지 여부에 대하여 이해할 수 있도록 돕는다.
- Metric 수집에 보다 자유로워 짐으로서, 시스템을 보다 폭넓게 이해할 수 있다. 메트릭을 추가하고 이를 추적하라!

### 2-1. Rule #1
- **Machine Learning이 없는 제품을 Launching하는 것에 두려워하지 말자.**
- Machine Learning은 반드시 데이터가 필요하므로 데이터가 없다면 사용하지 말자!
- Human Editing을 두려워하지말자.

### 2-2. Rule #2
- **Metric을 design하고 실행하라**
- Machine Learning이 수행할 것들을 공식화하기 전에, current system에서 가능한 많이 추적하라.
    - 초기에 시스템 사용자들로부터 보다 쉽게 permission을 얻을 수 있다. 
    - 추후에 걱정거리가 생길 것 같다면, 지금이라도 과거의 데이터를 얻는 것이 좋다.
    - metric instrumentation을 고려해서 시스템을 design한다면, 상황이 더 좋아질 것이다.
    - metric을 통해, 무엇이 변화하고 무엇이 그대로 유지되는 지 알게 될 것이다. 예를 들어, 환경의 급격한 변경은 초기에 메트릭을 크게 변화시키지 않을 수 있다.

### 2-3. Rule #3
- **complex heuristic 대신 Machine Learning을 선택하라**
- simple heuristic은 우리의 제품을 출시할 수 있게 해주지만, complex heuristic은 유지보수가 어려울 수 있다.
- 데이터가 있고, 성취하고자 하는 아이디어가 있다면 Machine Learning으로 한 번 생각해보자.
- heuristic이든 Machine Learning이든 우리는 끊임없이 update하기를 원할 것이고, Machine Learning을 활용하는 것이 update하기도 쉽고 유지하고 좀 더 쉽다는 것을 깨닫게 될 것이다.

## 3. ML phase 1: Your First Pipeline
- 우리들의 첫번째 pipeline을 deploying하는 것에 관하여 설명한다!
- 첫번째 pipeline을 위한 system infrastructure에 집중하라!
- 우리의 첫 pipeline을 신뢰하지 못하면, 어떤 일이 발생하는지 제대로 파악하기 어렵다.

### 3-1. Rule #4
- **첫번쨰 모델은 단순하게 유지하면서 인프라를 올바르게 구축하라!**
- 첫번째 모델이 우리의 제품에 있어서 빠른 성장을 불러오므로, 처음부터 화려할 필요는 없다.
- 우리가 기대했던 것보다더 훨씬 많이 인프라의 문제를 겪게 될 것이다.
- 모델을 화려하게 만들기 앞서 다음을 결정해야 한다.
    - Learning Algorithm으로부터 어떻게 examples를 얻을 것인가
    - 시스템에 '좋은'과 '나쁜'이 어떤 의미인가
    - 모델을 프로그램에 어떻게 통합시킬 것인가?
- 처음에 간단한 모델을 선택하는 것은 다음을 보다 쉽게 보장해준다.
    - features가 우리의 Learning Algorithm에 정확히 접근한다.
    - 모델은 합리적인 weights를 학습한다.
    - features는 서버의 모델에 정확히 도달한다.
- 위의 3가지를 갖췄다면, 대부분의 일을 다 한 것이다.
- 단순 모델로부터 더 복잡한 모델을 테스트할 수 있도록 baseline metrics를 제공한다.
- 첫번째 launching은 산만해지는 것을 피하기 위해 Machine Learning에서 얻어지는 가치를 낮춘채 Launching하는 것이다.

### 3-2. Rule #9
- **모델을 베포하기 전에 문제를 확인하라**
- 베포한 모델에 문제가 있을 경우, 이는 사용자가 직면할 문제이다.
- 모델을 베포하기 직전에 온전성을 체크하라.
- 특히, held out data에 대하여 모델의 성능이 합리적인지 확인하라!
- 데이터에 대하여 걱정이 있을 때는 모델을 베포하지 마라.
- 모델을 deploy하는 많은 팀들은 모델을 베포하기 전에 계속해서 RCO curve를 확인한다.
- 베포하지않은 모델에 대해서는 email 경고만 있으면 되지만, 베포하여 사용자가 직면한 문제들은 페이지를 수정해야할 것이다.
- 사용자에게 영향을 미치기 전에 기다리고 확신하는 것이 더 좋다.

### 3-3. Rule #12
- **directly하게 최적화하기 위해서 어떤 objective를 선택할 지 너무 고민하지 마라!**
- 수 많은 metric이 존재하고 우리는 그것을 전부 고려하고 측정해보아야 한다.
- 하지만, directly하게 최적화하지 않은 항목들도 학습 초기에는 모두 상승하는 것을 볼 수 있다.
- 그러므로, 모든 메트릭들이 쉽게 증가할 때, 메트릭들 간의 balance를 유지하려고 크게 고민할 필요없다. 간단하게 생각하라!
- 하지만, 너무 이 규칙에 대해서 많이 적용하지는 마라! 궁극적으로 원하는 시스템의 상태와 objective를 혼돈하지 마라!
- 만약, launch하지 않기로 결정하였는데 directly하게 최적화된 메트릭이 상승하고 있는 것을 발견한다면, 몇몇 objective의 수정이 필요할 수도 있다.

### 3-4. Rule #13
- **첫번째 objective를 위해서, 간단하고, 관찰가능하고, attributable한 metric을 선택하라!**
- 우리는 종종 진정한 objective가 무엇인지 모른다.
- 이전 ML system과 새로운 ML system 사이의 분석과 데이터를 확인하면서 objective를 수정할 필요성을 깨달을 것이다.
- 게다가, 다른 팀원들이 종종 진정한 objective에 동의하지 않을 수 있다.
- ML objective는 진정한 objective를 나타내야하고 쉽게 측정가능한 무언가여야만 한다!
- 사실, 진정한 objective가 없을 경우도 있다.
- 그래서 간단한 ML objective를 학습하고, 최종 ranking을 매기기 위한 추가적인 logic을 추가할 수 있도록 "policy layer"를 top layer에 두는 것을 고려하라!
- 모델이 하기 가장 쉬운 것은 직접적으로 관찰되고 시스템 동작에 기여하는 사용자의 동작이다. 다음 것들을 먼저 파악하자.
    - Was this ranked link clicked?
    - Was this ranked object downloaded?
    - Was this ranked object forwarded/replied to/emailed?
    - Was this ranked object rated?
    - Was this shown object marked as spam/pornography/offensive?   
- 간점적인 영향은 좋은 메트릭을 만들고, A/B 테스트와 launching을 결정하는 것에 활용될 수 있다. 하지만, 처음에는 모델링에 간접적인 것들은 피하도록 하자. 다음 것들을 처음에는 피할 수 있도록 하자.
    - Did the user visit the next day?
    - How long did the user visit the site?
    - What were the daily active users? 
- 마지막으로, ML에 있어서 다음 것들은 파악하려고 노력하지 마라.
    - Is the user happy using the product?
    - Is the user satisfied with the experience?
    - Is the product improving the user’s overall well-being?
    - How will this affect the company’s overall health?
- 위에서 언급한 것들은 모두 중요하지만, 믿을 수 없을 정도로 측정하기가 어렵다.
- 대신에 다음의 proxies들을 사용하도록 하자.
    -  if the user is happy, they will stay on the site longer. 
    -  If the user is satisfied, they will visit again tomorrow.
- well-being과 company health에 관하여, 우리가 팔고 계획중인 제품과 ML objective 사이의 연결에 있어서 인간의 판단이 요구될 수도 있다.

## 4. ML phase 2: Feature Engineering
- 이번 section에서는 파이프라인에 새로운 features를 추가하면서 어떻게 평가하고 training serving skew를 하는지 설명한다.
- ML lifecycle 첫번째 단계에서 중요한 것은 학습 데이터를 가져오고, 관심있는 metric을 가져오고, 전체적인 서비스 인프라를 구축하는 것이다.
- 테스트 할 system 및 unit을 가지고 있는 end to end system이 작동한 이후에 다음 두번째 단계가 시작된다.
- 두번째 단계에서는 가능한 한 많은 features를 끌어내고, 직관적으로 결합하는 것을 포함한다.
- 두번째 단계에서는 모든 metric이 상승해야하고, 많은 launching이 있을 것이다.
- 좋고 엔지니어들을 많이 끌어들일 수 있는 좋은 시기이다.

### 4-1. Rule #29
- **serving time에 사용된 feature들의 집합을 저장하고 이를 로그에 연결하여 학습 때 사용하는 것이 가장 좋다.**
- 모든 examples에 대하여 수행할 수 없을지라도, serving과 training 사이의 일관성을 확인하기 우해서 작은 비용을 투자하더라도 수행해보아랴.
- 이렇게 하면, serving time에 로깅 features로 전환하고 이와 함께 품질 개선이 이루어 졌으며 코드 복잡성 또한 줄었다. 현재 많은 팀들이 인프라 전환을 하고 있다. 

## 5. ML phase 3: Slowed Growh, Optimization Refinement, and Complex Models
- plautau에 도달했을 때, 어떤 것을 하면 좋을 지에 대하여 논의한다.
- 2단계에서 마무리에 도달했다는 것이 확실하게 느껴질 것이다. 예를 들면 한달 수익이 줄어들기 시작할 것이고 metrics들 사이에서 tradeoff가 발생할 것이다.
- 이루고자 하는 것이 점점 더 어려워지므로, Machine Learning은 점점 더 정교해져야 한다.
- 1, 2단계에서는 행복한 시간을 보냈다면 3단계에서는 각 팀들만의 길을 모색해야한다.

### 5-1. Rule #38
- **'unaligned' objective issue가 발생했다 하더라도 새로운 feature에 대해서 시간을 낭비하지 마라**
- measurements가 고점에 도달하게 되면, 현재 ML system의 objective를 벗어나는 문제들을 검토하기 시작한다.
- 앞서 언급한 것처럼, product goals가 존재하는 Algorithm objective에 의해서 다뤄지지 않는다면, objective 또는 product goals을 변경할 필요성이 있다.
- 예를 들어, 클릭이나 다운로드를 최적화할 수는 있지만, 출시를 결정하는 것은 human raters를 기준으로 해야할 수도 있다.

### 5-2. Rule #39
- **출시를 결정하는 것은 장기적인 제품 목표를 의미한다.**
- ML을 사용하여 직접 최적화하는 것은 여러 기준들에 의존한다.
- 제품의 건강 상태를 나타내는 'hit point'가 없으므로, 우리들이 수집한 데이터의 통계를 활용하여 앞으로 얼마나 좋을지 예측해야한다.
- Day Active users(1DAU, 30DAU), 수익률(revenue), 광고주들의 ROI들에 신경 써야 한다.
- A/B test와 같은 metrics들은 사용자 만족, 사용자 증가, 파트너 만족, 수익과 같은 장기적인 목표를 위한 proxy이지만, 한편으로는 고품질의 제품과 회사의 번창을 위한 proxy로 간주될 수도 있다.
- 즉, A/B test와 같은 것들이, 단순히 하나의 지표가 아니라 이들로부터 장기적인 현황을 알 수 있으므로 하는 것이 좋다!
- 가장 쉬운 결정은 모든 metric이 더 좋아지는 것이고 혹은 모두 나빠지고 있는 것이다.
- 각각의 metric들은 각 팀들이 우려하고 있는 위험 부담을 포함하고 있다.
- 팀의 궁극적인 관심사인 '앞으로 5년 후 제품이 어떻게 될 것인가?"에 대한 metric은 없다.





