---
layout: post
title: "MLOps 개론"
categories: boostcamp
tags: main
comments: true
---
MLOps에 대한 전반적인 내용을 학습한다.

## 목차
- [1. MLOps가 필요한 이유](#1-mlops가-필요한-이유)
- [2. MLOps Component 이해](#2-mlops-component-이해)
- [3. 기타 논의](#3-기타-논의)

## 1. MLOps가 필요한 이유
- 반복적인 업무를 자동화 시킬 수 있다.
- 머신러닝/딥러닝을 적용한 비즈니스 가치를 창출할 수 있다.
- **Production환경과 Research환경에서의 차이로부터 발생하는 Risk를 줄이기 위함**
    - Production환경과 REsearch환경은 다소 차이를 보인다.
![image](https://user-images.githubusercontent.com/53552847/144781434-c4efc9a0-c4c8-4fbe-ad3d-b4f21880e12b.png)
    - 이러한 차이로부터 생기는 문제를 최소화하고자 한다.

## 2. MLOps Component 이해
### 2-1. MLOps Component의 종류   
- 인프라 (서버, GPU)
- Serving
- Experiment, Model Management
- Feature Store
- Data Validation (Data Management)
- Continuous Training
- Monitoring
- AutoML

### 2-2. 인프라 (서버, GPU)
- **Production을 베포하기 위해서 어느 정도의 인프라가 필요한지에 대한 search가 필요하다.**
- 예상 트래픽이 어느정도 되는지?!
- 이런 트래픽에 대하여 서버의 CPU, Memory는 어느정도로 할 것인지?!
- 혹여나 사이트가 인기가 많아져서 트래픽이 많아지거나, 혹은 트래픽이 적어질 때, 스케일 업, 스케일 아웃이 가능한지?!
- 자체 서버를 구축하여 Production 베포를 할 것인지 혹은 클라우드를 활용할 것인지?!

### 2-3. Serving
- **우리의 제품을 고객들이 활용할 수 있게 전달해주어야 한다. 이 때, 어떤 방법으로 전달을 하는 것이 보다 효율적인가?!**
- 일정 시간마다 Serving을 받기 희망하는 Batch Serving과 실시가능로 받기를 희망하는 Online Serving이 있다.
- Batch Serving은 많은 양을 일정 주기로 서빙하는 것이고, Online Serving은 한 번에 하나씩 실시간으로 서빙하는 것이라고 이해할 수 있다.
- 특히, Online Serving의 경우, 갑작스런 트래픽 증가가 있을 수 있으므로, 병목 현상이 일어나지 않게 대비해야하고, 확장 가능하도록 준비해야 한다.
![2](https://user-images.githubusercontent.com/53552847/144781520-ebb20da4-9e5e-408c-bdb7-d1f5a8ff4237.jpg)


### 2-4. Experiment, Model Management
- **ML/DL을 실험하는 과정에서, 어떤 모델이 더 좋은지, 어떤 파라미터로 학습 했을 때 성능이 더 좋았는지, inference 속도가 더 빨랐는지 등을 기록하는 것은 정말 중요하다.**
- 모델 Artifact, 모델 성능, HyperParameter를 기록하고 좋은 모델을 Production에서 활용한다.
![3](https://user-images.githubusercontent.com/53552847/144781606-48e0f6c9-2d8d-4e1a-9fba-27ba3b00b21f.jpg)

### 2-5. Feature Store
- **데이터를 전처리와 같은 반복적인 업무를 할 때, 불필요한 시간이 할애되므로, 이를 미리 만들어 놓고 재사용한다면 보다 편리하고 빠르게 이후 업무를 진행할 수 있다.**
- Feature Store의 경우, Deep Learning에서는 많이 발전하지 않았고, 정형 데이터에서 특히 많이 발전하였다.
- 현재, library를 활용하기 보다는 직접 개발하여 활용하는 것이 더 많다.

### 2-6. Data Validation
- **Research에서 활용한 데이터 및 feature와 Production에서 활용하는 데이터 및 feature가 유사한지 확인할 필요성이 있다.**
    - Research의 경우, 고정된 데이터 및 Outlier 처리 등을 진행한다.
    - Production은 제품을 활용하는 고객들로부터 얻는 데이터가 있을 뿐만 아니라, 그 중에서는 outlier들이 포함되있을 수도 있으므로, Research 딴에서의 데이터보다 보다 저품질일 확률이 높다.
- 위의 이유로부터, Data Drift, Model Drift, Concept Drift가 발생한다.
- Drift에 대한 내용은 다음의 링크를 참고하도록 하자.
[How to Detect Model Drift in MLOps Monitoring](https://towardsdatascience.com/how-to-detect-model-drift-in-mlops-monitoring-7a039c22eaf9)
[Productionizing Machine Learning: From Deployment to Drift Detection](https://databricks.com/blog/2019/09/18/productionizing-machine-learning-from-deployment-to-drift-detection.html)

### 2-7. Continuous Training
- **활용하고 있던 모델을 중간에 다시 학습해야하는 경우가 있다. 특히, 트렌드는 짧은 시간 내에 바뀌기를 반복하므로 이러한 정보를 얻고 학습하고를 반복할 필요성이 있다.**
- Product를 활용하는 고객으로부터 얻어지는 새로운 데이터를 모델에 재학습하기 위해!
- Metric 기반으로 성능이 떨어졌을 경우, 다시 높이기 위해 재학습을 진행한다.

### 2-8. Monitoring
- **Research에서 발견한 Model에 Production에서도 잘 작동하는지를 기록해야한다.**
- 모델의 지표가 어떻게 변하는지 확인한다.
- 인프라의 성능이 어떻게 변화하고 있는지 확인한다.

### 2-9. AutoML
- **사람이 하나 하나씩 모델을 집어넣어 학습을 진행하기에는 소모되는 비용 대비 얻어지는 이득이 적다.**
- 이를 AutoML로 대채함으로서 Research, Production에서 모두 이용하고, 어떤 차이가 있고 어떤 모델이 성능이 좋은지 자동으로 학습 및 확인할 수 있다.

### 3. 기타 논의
- 매력적으로 생각하는 MLOps Component TOP3
    - AutoML: 사람이 하는 일을 자동으로 처리해줌으로서 일의 유연성을 늘릴 수 있다.
    - Continuous Training: Production이란 고객들의 수요, 트렌드에 예민하다. 이를 보다 잘 모델에 반영하기 위하여 Continuous Training은 보다 중요한 Component라고 생각한다.
    - Serving: 특히 Online Serving의 경우, "빨리 빨리"를 강조하는 현대 사회에서, 굉장히 중요한 부분이라고 생각하며 이를 잘 해내는 것은 선택이 아닌 필수라고 생각한다.
