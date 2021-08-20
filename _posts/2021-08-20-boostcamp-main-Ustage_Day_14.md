---
layout: post
title: "Ustage Day 14"
categories: boostcamp
tags: main
comments: true
---
PyTorch 활용법에 대하여 학습한다.

**부스트 캠프 14일차 학습 요약**
- **행사** : 스페셜 피어세션, 오피스아워
- **학습** : Multi-GPU 학습, Hyperparameter Tuning, PyTorch Troubleshooting
- **피어 세션**

## 목차
- [1. Multi-GPU 학습](#1-multi--gpu-학습)
- [2. Hyper-Parameter Tuning](#2-hyper-parameter-tuning)
- [3. PyTorch TroubleShooting](#3-pytorch-troubleshooting)
- [4. 피어 세션](#4-피어-세션)
- [5. 14일차 후기](#5-14일차-후기)
- [6. 해야할 일](#6-해야할-일)

## 1. Multi-GPU 학습
- 딥러닝 모델을 병렬화 하는 Model Parallel의 개념을 학습한다.
- 데이터 로딩을 병렬화 하는 Data Paralle 개념을 학습한다.
- 딥러닝 학습 시에 GPU가 동작하는 프로세스에 대한 개념을 학습한다.
- 어떻게 GPU를 다룰 것인가??

### 1-1. 개념 정리
- Single vs Multi
    - Single은 한 개, Multi는 두 개
- GPU vs Node(or System)
    - Node는 한 대의 컴퓨터를 의미한다.
    - Single Node Single GPU -> 한 대의 컴퓨터에 한 개의 GPU
    - Single Node Multi GPU -> 한 대의 컴퓨터에 여러개의 GPU
    - Multi Node Multi GPU -> 여러대의 컴퓨터에 여러개의 GPU

### 1-2. Model Parallel (모델의 병렬화)
- 다중 GPU 학습을 분산하는 두 가지 방법 -> 모델 나누기 & 데이터 나누기
- 모델을 나누는 것은 생각보다 예전부터 사용했다. (AlexNet)
- 모델의 병목, 파이프라인의 어려움 등으로 인해 모델 병렬화는 고난이도 과제이며, 흔히 사용되지는 않는다고 한다.
- Model Parallel의 대한 코드는 다음과 같다.
![pytorch 18](https://user-images.githubusercontent.com/53552847/130241802-1f19de20-abfe-4828-ae20-1a3879ee0347.PNG)

### 1-3. Data Parallel (데이터 로딩을 병렬화)
- 데이터를 나눠 GPU에 할당한 후 결과의 평균을 취하는 방법이다.
- minibatch 수식과 유사한데 한 번에 여러 GPU에서 수행한다.
- PyTorch에서 다음의 두 가지 방식을 제공한다.
    - DataParallel : 단순히 데이터를 분배한 후 평균을 취함. (GPU 사용 불균형 문제가 발생하며, Batch 사이즈 감소 (여러 GPU로부터 데이터를 취합하는 GPU에서 터지는 현상이 발생하기 때문이다.), GIL
    - DistributedDataParallel : 각 CPU 마다 Process를 생성하여 개별 GPU에 할당 (기본적으로 DataParallel로 각각 개별적으로 연산의 평균을 낸다.)
        - 데이터를 취합한 후에 평균을 내는 것이 아니라 각각의 평균을 낸 후에 취합한다.
    - DataParallel의 코드는 다음과 같다.
![pytorch 19](https://user-images.githubusercontent.com/53552847/130241808-2c881475-0dbf-491e-aa28-be9b6a2adebd.PNG)
    - DistributedDataParallel의 코드는 다음과 같다.
![pytorch 20](https://user-images.githubusercontent.com/53552847/130241809-81bf60c6-1586-4018-a126-9b7115f47240.PNG)
    - 위의 코드에서 각 변수는 다음의 의미를 띈다.
        - pin_memory : DRAM, 즉 메모리에 데이터를 바로바로 올릴 수 있도록 절차를 간소하게 데이터를 저장하는 방식이다. 메모리에 데이터를 올린 다음 이것이 GPU로 가서 연산되게 되는데 이 과정을 빠르게 해준다.
    - DistributedDataParallel의 메인함수는 다음과 같다.
![pytorch 21](https://user-images.githubusercontent.com/53552847/130241811-4b406f19-f73a-4d3c-a78a-156afa99d859.PNG)
    - DistributedDataParallel이 DataParallel 보다 약간 더 귀찮은 과정이 있다는 정도로만 이해하자.

## 2. Hyper-Parameter Tuning
- Ray Tune Framework을 이용해 최적화하는 방법을 학습한다.
- Grid & Random, Bayesian과 같은 기본적인 Parameter Search 방법론들과 Ray Tune 모듈을 사용하여 PyTorch 딥러닝 프로젝트 코드 구성을 익힌다.
- 어떻게 Hyper-Parameter를 잘 조정해서 더 좋은 성능을 낼 수 있을까?

일반적으로 결과가 잘 안나왔을 경우, 다음과 같은 것들을 할 수 있다.
- 모델을 바꾼다. (하지만, 이미 대부분의 좋은 모델들이 고정되어 있다. 이러한 이유로 모델로부터 얻을 수 있는 이익이 그렇게 크지만은 않다.)
- 데이터를 바꾼다. 새로운 데이터를 추가하거나, 기존 데이터의 오류를 찾는 방법이다.
    - 일반적으로 가장 중요한 방법이다.
    - Data는 다다익선
- Hyper-Parameter Tuning
    - Hyper-Parameter Tuning으로 부터 얻을 수 있는 이득은 그리 크지많은 않다.
    - 마른 수건도 짜보라는 마인드로 진행해야 한다.

### 2-1. Hyper-Parameter Tuning
- 모델 스스로 학습하지 않은 값은 사람이 지정한다.
    - Learning Rate(NAS 모델?), 모델의 크기, Optimizer 등
    - Hyper Parameter에 의해서 값이 크게 좌우될 경우도 있지만, 요즘에는 많이 줄어들었다.
    - 마지막 0.01을 쥐어짜야할 때 도전할만 하다.
    - 가장 기본적인 방법이 Grid Search와 Randomized Search인데, 각각은 다음과 같으며, 요즘에는 더 이상 쓰지 않는다고 하니 개념만 알고 넘어가자.
        - Grid Search : 각각의 Hyper Parameter들을 일정 기준으로 미리 정해놓고 이 각각에 대하여 전부 실행해본 후 가장 결과가 좋은 것을 선택하는 방법니다.
        - Randomized Search : Hyper Parameter의 값들을 랜덤하게 선택하여 돌려본 후 그 중 가장 성능이 좋은 것을 선택한다.
        - 흔히, 처음에는 Randomized Search를 활용해 진행하다가 성능이 어느 정도 잘 나오는 구간을 찾으면 그 구간에 대해서만 Grid Search를 진행하는 방식으로 진행한다. -> But, 옛날 방식이다.
    - 요즘에는 Bayesian 기법들이 주로 구성되어 있다. (BOHB 논문을 참고하면 도움이 된다.)

### 2-2. Ray (알아두면 아주 좋으며, 관심을 가지고 알아보도록 하자.)
- Multi-Node Multi Processing 지원 모듈
- ML/DL의 병렬 처리를 위해 개발된 모듈
- 기본적으로 현재의 분산 병렬 ML/DL 모듈의 표준이다.
- Hyper-Parameter Search를 위한 다양한 모듈을 제공한다.
- Ray의 코드는 일반적으로 다음과 같다.
![pytorch 22](https://user-images.githubusercontent.com/53552847/130241813-25d84e55-75fb-4acd-84c8-82d8ed79cef4.PNG)
- 위의 코드에서의 변수는 다음의 의미를 가진다.
    - ASHAScheduler : 일반적으로 알고리즘이 실행이 되면서 중간중간에 우리가 의미 없다고 생각하는(Loss값이 잘 나오지 않는) Metric을 잘라내는 알고리즘이다.
    - CLIReporter : 결과를 출력하는 양식이다.
    - tune.run : 병렬 처리 양식이다.
        - partial : 데이터를 쪼갠다.
        - resources_per_trial : 한 번 trial 할 떄, 쓸 수 있는 GPU, CPU의 개수
        - config : search space
    - Scheduler를 잘 사용해주면 굉장히 효율적으로 학습할 수 있다.
    - Bayesian Optimizer와 같은 알고리즘을 선택할 수 있는데, 이러한 좋은 알고리즘을 사용하면 굉장히 효율적인 Hyper-Parameter Tuning을 할 수 있다.
    - Ray에서는 우리의 코드가 하나의 함수로서 입력이 되어야지만, 나중에 불러올 수 있다. (즉, Hyper-Parameter Tuning, Ray를 할 때에는 항상 함수의 형태로 저장하도록 하자!)
    
## 3. PyTorch Troubleshooting
- PyTorch를 사용하면서 자주 발생할 수 있는 GPU에서의 Out Of Memory 에러 상황들의 예시를 보고 해결한다.
- FAQ와 같은 느낌으로 다음의 내용을 읽을 수 있다.

### 3-1. OOM이 해결하기 어려운 이유
- 어디서, 왜 발생했는지 알기 어렵다.
- Error Backtracking이 이상한 곳으로 간다.
- 메모리의 이전 상황의 파악이 어렵다.
- 보통 iteration을 돌면서 특정 시점에 에러가 발생하는 경우가 많다.
- 가장 단수하고 1차원적인 해결책은 Batch size down -> GPU clean -> Run 이다.

### 3-2. OOM 해결할 수 있는 방법
#### GPUUtil 사용하기
- nvidia-smi처럼 GPU의 상태를 보여주는 모듈이다.
- Colab 환경에서 GPU 상태를 보여주는 것이 편하다.
- iteration마다 메모리가 늘어나는지 확인할 수 있다.
- GPUUtil의 코드는 다음과 같다.
![pytorch 23](https://user-images.githubusercontent.com/53552847/130241814-976a296d-5de1-47ac-8bad-5fae7885e4d4.PNG)
- 만약, iteration이 돌면서, Memory가 지속적으로 늘어나다면 흔히, 어딘가에 메모리가 잘못 쌓이고 있다는 것으로 파악할 수 있다.

#### torch.cuda.empty_cache() 사용하기
- 사용되지 않은 GPU 상 cache를 정리한다.
- 가용 메모리를 확보한다.
- del과는 구분되며, 커널을 다시 돌리는 방법 대신 사용하기 좋은 함수 이다.
- torch.cuda.empty_cache()에 대한 코드는 다음과 같다.
![pytorch 24](https://user-images.githubusercontent.com/53552847/130241816-b723af0a-a990-4381-8d19-c46fc03faf79.PNG)
- 위의 코드를 보면 알 수 있듯이, del을 한다고 해서 바로 memory를 사용할 수 있는 것이 아니라 garbage collector를 사용해야지 완전하게 memory를 사용할 수 있다. 즉, gc를 강제하는 것을 torch.cuda.empty_cache()이 한다고 볼 수 있다.
- 학습 전에는 꼭 `torch.cuda.empty_cache()`를 사용하는 것을 권장한다.

#### training loop에 tensor로 축적되는 변수는 확인하자!
- tensor로 처리된 변수는 GPU 상에 메모리로 사용한다.
- 해당 변수 loop 안에 연산이 있을 경우, GPU에 computatioknal graph를 생성한다. (즉, 메모리 잠식이 일어난다.)
- 1번만 사용하는 변수(즉, 1-d tensor)의 경우, python 기본 객체로 변환하여 처리하도록 하자.

#### del 명령어 적절히 사용하기
- 필요가 없어진 변수는 적절한 삭제가 필요하다.
- python의 메모리 배치 특성상 loop이 끝나도 메모리를 차지한다.

#### 가능한 batch 사이즈 실험해보기
- 학습시 OOM이 발생했다면 batch 사이즈를 1로 해서 실험해보기
- 다음의 코드의 방식으로 실험할 수 있다.
![pytorch 25](https://user-images.githubusercontent.com/53552847/130241818-0ebe1057-7ffa-4a53-b395-1a12005a1c2c.PNG)
- 위와 같이 코드를 사용하게 되면, 코드가 조금 더 정갈하게 되며, batch_size를 바로 1로 하는 것이 아니라, 어느 정도 실험을 하다가 1로 바꿀 수 있다는 장점이 있다.
- OOM이 발생했을 때, batch_size가 어디까지 가능한지 확인하기 위해서 위와 같은 코드를 구성할 수 있다.

#### torch.no_grad() 사용하기
- Inference 시점에서는 torch.no_grad() 구문을 사용한다. Inference 시점에서는 반드시 사용하도록 하자!
- backward가 일어나지 않는다.
- backward pass으로 인해 쌓이는 메모리에서 자유롭다.

#### 예상치 못한 에러 메세지
- OOM 말고도 유사한 에러들이 발생한다.
- CUDNN_STATUS_NOT_INIT 혹은 device-side-assert 등
    - 전자는 GPU를 잘못 설치했거나, 제대로 안 설치했을 경우 생기는 오류이다.
    - 후자는 OOM의 일종으로 볼 수 있다.
- 해당 에러도 cuda와 관련하여 OOM의 일종으로 생각될 수 있으며, 적절한 코드 처리가 필요하다.
- https://brstar96.github.io/shoveling/device_error_summary/ 를 참고해서 OOM이 발생했을 경우, 참고하도록 하자.

#### 그 외
- colab에서는 너무 큰 사이즈는 실행하지 말 것
    - linear, CNN, 특히 LSTM 등
    - 그냥 Pre-Trained Model을 사용하는 것이 일반적으로 성능이 더 좋다.
- CNN의 대부분의 에러는 크기가 안 맞아서 생기는 경우이다.
    - torchsummary 등을 활용해서 사이즈를 맞춰 주도록 하자.
- tensor의 float.precision을 16bit로 줄일 수 있다.
    - 많이 쓰진 않는다.
    - 엄청 큰 모델을 돌릴 경우에 많이 사용한다. 현재는 많이 사용할 일이 없다.

## 4. 피어 세션
### 4-1. [팀 회고록 정리]
각자 회고록을 조금의 시간을 갖고 작성했다.
각자 잘했던것, 아쉬운것, 도전할것을 발표하는 시간을 가졌다.

### 4-2. [스페셜 피어세션]
다른 팀들 중 참고할만한 것 - 하루의 TODO 리스트를 만들어 각자 slack에 공유하기

### 4-3.[그외에 한일]
다음주 목요일까지 ViT 논문을 읽고 피어세션과 멘토링 시간에 리뷰하기
CV, NLP 도메인에 대한 이야기를 깊게 나누었다.

## 5. 14일차 후기
어제까지 해서 많고 어려웠던 과제가 마무리 되서 그런지 오늘은 나름 쉬어가는 하루였던 것 같다. 강의 역시 어려운 내용이 아니라 코딩을하면서 나름의 꿀팁들을 알려준 내용이었기에 조금은 편안하다는 생각이 들었다.

스페셜 피어세션을 통해서 다른 팀원들과의 교류도 해보고, 피어세션에서는 다음주부터 있을 P-stage에 대하여 이야기를 나누며, 담소를 조금 나누는 시간도 가졌다. 한 층 더, 팀원들과 가까워진 것 같다는 생각이 들어서 좋았던 시간이었던 것 같다.

마지막으로, 오피스 아워를 통해 이번 주 과제에 대한 설명을 들었는데, 우리가 어려워 한게 당연할 만큼 난이도가 있었다고 말씀해주셔서 조금은 위로가 되기도 했고, 멘토님들의 꿀강의를 들으며 한 층 더 이해의 폭이 넓어진 것 같다.

## 6. 해야할 일
- pin_memory를 사용했을 때, 메모리에 데이터를 올린 다음 이것이 GPU로 가서 연산되게 하는데 이 과정을 빠르게 한다면 True로 Default를 주는 것이 좋지 않을까?
- 모델의 모든 layer에서 learning rate가 항상 같아야 하는가?
- ray tune을 이용해 Hyper-Parameter Search를 하려고 한다. 이 때, 아직 어떤 Hyper-POarameter도 탐색한 적이 없지만 시간이 없어서 1개의 Hyper-Parameter만 탐색할 수 있다면, 어떤 Hyper-Parameter를 선택하는 것이 좋을까?
- NAS란 무엇인가?
- BOHB 논문을 읽어보자.
- Ray 표준문서를 통해 함수들을 확인해보자.
- python memory 관리에 대하여 학습하도록 하자.
