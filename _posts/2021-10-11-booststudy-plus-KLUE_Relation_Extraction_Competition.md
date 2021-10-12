---
layout: post
title: "KLUE Relation Extraction Competition"
categories: booststudy
tags: plus
comments: true
---
KLUE Relation Extraction 대회(In BoostCamp) 회고

**AUPRC score : 1st place out of 19 teams.**  
**Micro F1 Score : 3rd place out of 19 teams.**

부스트캠프에서 2주간 진행한 KLUE Relation Extraction 대회 진행과정에 대한 설명을 진행합니다! 본 과정은 Boostcamp HappyFace팀이 함께 진행하였습니다.

## 목차
- [1. Relation Extraction이란?](#1-relation-extractio이란)
- [2. Dataset](#2-dataset)
- [3. EDA](#3-eda)
- [4. Dataset 구축](#4-dataset-구축)
- [5. Tokenizer](#5-tokenizer)
- [6. Data Augmentation](#6-data-augmentation)
- [7. Model](#7-model)
- [8. Something Importance](#8-something-importance)
- [9. Have to Try](#9-have-to-try)
- [10. 협업](#10-협업)

## 1. Relation Extreaction이란?
- **하나의 문장이 있을 때, 특정 단어(entity)들 사이의 속성과 관계를 추론하는 문제**
- 개체명 인식을 통해 entity에 대한 정보를 찾고, 관계 추출을 통해 그 두 개의 entity 사이의 관계를 출력해낸다.
- **RE를 하는 이유는?**
    - 문장 속에서 단어 사이의 관계를 파악함으로서 주어진 문장을 한결 더 부드럽게 이해할 수 있다.
    - 대규모 비정형 텍스트로부터 자동화된 정보를 수집할 수 있다.
    - 언어로부터의 핵심 정보 추출 및 요약이 가능하다.
- 활용 범위
    - QA 시스템
    - 지식 그래프 구축
- 예시
    -  "\<something\>는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다." 라는 문장이 있고, "조지 해리슨", "비틀즈"가 있을 때 이 둘 사이의 관계를 구하는 문제이다.
    -  위의 문제에서는 no_relation이 관계에서의 정답이다.

## 2. Dataset
- Dataset 통게
    - train dataset : 총 32470개
    - test dataset : 7765개 (label은 전부 100으로 처리되어 있다.)
- Data 예시
    - id, sentence, subject_entity, object_entity, label, source로 구성된 csv 파일
    - sentence example : \<Something\>는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다. (문장)
    - subject_entity example : {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'} (단어, 시작 idx, 끝 idx, 타입)
    - object_entity example : {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} (단어, 시작 idx, 끝 idx, 타입)
    - label example : no_relation (관계),
    - source example : wikipedia (출처)
- class에 대한 정보는 다음과 같다. 
![1](https://user-images.githubusercontent.com/53552847/136720424-47bba972-ad92-4ddb-9d80-71bbfd4345fb.PNG)

## 3. EDA
**Maximum token length**를 세팅, **Imbalanced Dataset**임을 확인
### 3-1. Token 개수 파악
-  tokenizer : `klue/bert-base` wordpiece tokenizer 활용
-  train/test dataset에서의 token 파악
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
||train|test|
|Data 개수|32470|7765|
|Unique token 개수|24478|19589|
|Max token 개수|229|221|
|Min token 개수|7|10|
|Mean token 개수|51|49|
|Medain token 개수|46|45|
|분포|![2](https://user-images.githubusercontent.com/53552847/136720425-190af853-087d-45e4-915e-8a23e0955ba4.PNG)|![3](https://user-images.githubusercontent.com/53552847/136720426-327084a2-cd99-4b26-8833-987b35ed4a6a.PNG)|
{: .tablelines}
- train / test에서의 비슷한 분포 확인
- train, test에서 비슷한 분포가 아닐 경우, train에서 학습하여 높은 성능이 나온다고 해도 test에서 성능이 떨어질 수 있기 때문이다.
    
![4](https://user-images.githubusercontent.com/53552847/136720417-634c2a21-3ab0-4d7a-8435-698e37a35b86.PNG)

### 3-2. Label별 Token 개수 파악
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|Label 0|Label 1|Label 2|...|
|![5](https://user-images.githubusercontent.com/53552847/136743812-11879fac-c474-4683-b6f3-21eb5e157882.PNG)|![6](https://user-images.githubusercontent.com/53552847/136720420-bc3e849d-f92c-4ffd-938d-c751aa070985.PNG)|![7](https://user-images.githubusercontent.com/53552847/136720421-dd861673-9995-4124-b4e2-ec27a9f96f90.PNG)||
{: .tablelines}
- 전체적으로 모든 class에서 비슷한 양상의 히스토그램을 띄는 것을 볼 수 있었다.
- tokenizer의 max_length의 경우 (train:229/test:221)이므로 이보다 짧게 구성해도 좋을 것이라고 판단.
- 특히 100~150 구간에 소수의 데이터가 분포하고 있으며, 128개의 토큰보다 더 많은 토큰을 가진 문장의 개수는 Train:1.4%/Test:0.8% 이므로 128 정도의 tokenizer의 max_length를 가지면 좋을 것이라고 판단하여 **max_token_length를 128로 선택**하였다.

### 3-3. Imbalanced Dataset
![71](https://user-images.githubusercontent.com/53552847/136720423-3d1e6616-8c8a-4e89-b922-59e365f93779.PNG)
- 0(= no_relation)이 대부분을 차지하고 있음을 확인할 수 있다.
- label별 분포의 차이가 꽤 크다는 것을 알 수 있다.
- 이를 토대로, Focal Loss 적용, Stratified KFold, Data Augmentation, Duo Classifier를 실험해보고자 했다.

## 4. Dataset 구축
- Relation Extraction의 경우, **각 단어가 subject인지, object인지 알려주는 것이 효과적인 것**으로 알려져 있으며, 이에 더불어 **subject이면서 person인지, 혹은 organization인지 등도 알려주면 더욱 효과적**임으로 여러 paper를 통해 확인할 수 있었다.([paper](https://arxiv.org/pdf/2102.01373.pdf)에서 대략적으로 확인할 수 있다.)

![8](https://user-images.githubusercontent.com/53552847/136720610-80d09c0a-9237-43db-96d5-535a8d9cc1fc.PNG)

- 위의 표에서 'Entity marker', 'Typed entity marker', 'Typed entity marker (punct)'를 실험하였다.
- 베이스라인 모델인 `klue/bert-base`에서 'Typed entity marker'의 성능이 가장 좋음을 파악할 수 있었고, 위의 표에 RoBERTa의 경우 'Typed entity marker (punct)'에서 가장 좋은 성능을 나타냄을 파악할 수 있었다.
- **'Typed entity marker (punct)'의 경우, `klue/roberta-large` tokenizer를 사용하였는데, 이 tokenizer의 영어 vocab sets은 단지 800개에 불과했고 이에 대하여 entity tag를 한국어로 교체한 후 실험을 진행하였다. ("@\*person\*조지 해리슨@" -> "@\*사람\*조지 해리슨@")**
- 더불어, **`klue/roberta-large`의 wordpiece tokenizer prefix의 경우 '##'으로 구성되어 있음을 고려하여, 'Typed enbtity marker (punct)'를 적용할 때, '#'대신 '&'를 적용하여 활용하였다.("안녕하세요" -> "안녕", "##하세요")**

- 이에, **'Typed entity marker', 'Typed entity marker (punct)'(kor version)을 활용하여 실험을 진행**하였으며, 각각의 예시는 다음과 같다.
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
||Sentence|
|Typed Entity Marker|\<Something\>는 [OBJ-PERSON]조지 해리슨[/OBJ-PERSON]이 쓰고 [SUB-PERSON]비틀즈[/SUB-PERSON]가 1969년 앨범 《Abbey Road》에 담은 노래다.|
|Typed Entity Marker Punct (kor version)|\<Something\>는 &^^조지 해리슨#이 쓰고 @\*사람\*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.|
{: .tablelines}

## 5. Tokenizer
- [3. EDA](https://jjonhwa.github.io/booststudy/2021/10/11/booststudy-plus-KLUE_Relation_Extraction_Competition/#3-eda)을 바탕으로 tokenizer의 **max_token_length의 경우 128**로 맞춰주었다.
- [4. Dataset 구축](https://jjonhwa.github.io/booststudy/2021/10/11/booststudy-plus-KLUE_Relation_Extraction_Competition/#4-dataset-%EA%B5%AC%EC%B6%95)에서 활용한 entity marker를 붙이는 작업을 진행할 경우, subject_entity, object_entity 각각 앞뒤로 1개 token이 추가되 총 4개의 token이 증가하게 되는데 이를 token_length에 반영해주는 것이 중요하다. 즉 **max_token_length에 +4**를 해준다.
- **Dynamic Padding**을 활용하기 위하여, tokenizer의 padding 옵션을 True가 아닌 max_length로 주어 적절히 max_length가 잘려서 불필요한 길이가 남아있지 않게 만들어 주었다.
- Uniform Length Batching 전략을 활용하여 길이가 유사한 문장들끼리 묶어서 진행하는 방법도 구현하고자 하였지만 2주라는 짧은 시간 관계상 직접 적용은 하지 못하였다. 이에 대한 내용과 Batching 전략이 중요한 이유에 대한 내용은 [snoop2head's log](https://snoop2head.github.io/Relation-Extraction/#setting-maximum-token-length-for-roberta-tokenizer)에서 확인할 수 있다.

## 6. Data Augmentation
- **EDA(Easy Data Augmentation)**
    - [EDA paper](https://arxiv.org/pdf/1901.11196.pdf) 에서 자세한 내용을 확인할 수 있다.
    - Random Insertion, Random Deletion, Random Swap, Synonym Replacement를 적용하였지만 유의미한 성능의 개선이 이루어지지 않아 적용하지 않았다.
- **Round-trip Translation**
    -  여러 언어로 번역한 후 다시 본래의 언어로 번역하여 같은 의미이지만 다른 형태를 띄게 만드는 Data Augmentation 기법이다.
    -  이를 Pororo NER translation 함수를 활용하여 적용하였다.
    -  2주라는 시간 관계상 약 5000개의 데이터 증가를 하였지만, 유의미한 성능 개선이 이루어지지 않아 적용하지 않았다.

## 7. Model
### 7-1. pre-trained model 선정
- Relation Extraction 부분에 대한 Model Benchmark를 토대로 Pre-trained Model을 선정하였다.
- Relation Extraction 부분이 없을 경우, Named Entity Recognition과 Natural Language Inference를 토대로 선정하였다.
- Benchmark
    - [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE#baseline-scores), [Tunib-Electra Benchmark](https://github.com/tunib-ai/tunib-electra), [KoElectra Benchmar](https://github.com/monologg/KoELECTRA)
    - 위의 Benchmark를 토대로 pre-trained model을 선정하였으며, 이를 토대로 **KoElectra, XLM-R-large, KLUE-RoBERTa-large**를 활용하여 실험하였다.
- KoElectra tokenizer의 경우 Unknown token의 발생이 성능 하락의 원인이 됨을 확인할 수 있었고 이를 바탕으로 unknown token이 가장 적게 발생하는 모델을 찾았을 때 XLM-R-large, KLUE-RoBERTa-large가 최종 실험 모델로 선정되었다.
- 같은 조건 하에서 XLM-R-large와 KLUE-RoBERTa-large를 실험하였을 때 다음과 같았고 최종적으로 **KLUE-RoBERTa-large를 backbone 모델로 선정**하였다.
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
||micro f1|auprc|
|XLM-R-large|70.525|76.537|
|KLUE-RoBERTa-large|73.349|80.332|
{: .tablelines}

### 7-2. Fine-tuning 전략
- [Relation Extraction on TACRED](https://paperswithcode.com/sota/relation-extraction-on-tacred)에서의 SOTA 모델 순서대로 적용하기(아래의 정보를 함께 고려하며 선택하였다.)
- Hidden feature 정보 활용
![9](https://user-images.githubusercontent.com/53552847/136721176-cbd96c85-be42-4d6c-9936-9cdb992044f7.PNG)
    - [Matching the Blanks: Distributional Similarity for Relation Learning](https://aclanthology.org/P19-1279.pdf)을 참고하여 선택하였다.
    - 간략하게 요약하자면, **classification을 진행할 때, [CLS] token만 활용하는 것이 아닌 정보를 많이 담고 있는 다른 token의 hidden state vector까지 활용하여 classification을 진행하는 것**이다.
    - 이를 바탕으로 어떤 토큰의 정보를 추가적으로 활용할지 선택하여 모델을 선정하였다.
    - 위의 6개의 모델에 대한 성능은 아래의 표와 같다.
![10](https://user-images.githubusercontent.com/53552847/136721443-04523e5e-fe40-4cbf-8983-a84c35f82057.PNG)
- 이에 대하여, **"(f)ENTITY MARKERS - ENTITY START"의 형태를 띄는 "RE Improved Baseline"과 "(e)ENTITY MARKERS - MENTION POOL"의 형태를 띄는 "RBERT"을 최종적으로 선정하여 진행**하였다.
- 더불어, **기존의 **"KLUE/RoBERTa-large"**를 바탕으로한 모델의 성능과 1~2점 정도의 차이밖에 없어서 기존 모델도 함께 사용하여 최종 Ensemble을 진행**하였다.

### 7-3. 모델 구조 및 Score
#### Customized Improved Baseline Model
![12](https://user-images.githubusercontent.com/53552847/136745002-2a451754-d863-4b8c-a177-ed07d53caf80.png)
- 구조
    - Backbone model : `klue/roberta-large`
    - typed entity marker (punct) (ver. kor) : "... @\*사람\*조지해리슨@ ... &^사람^비틀즈& ..."
    - 기존의 Improved Baseline paper에서 entity tag를 감싸는 marker로 "#"을 사용했지만 'klue/roberta-large' tokenizer의 prefix와 겹치기 때문에, 이로 인한 문제 발생을 방지하기 위하여 '#' -> '&'을 사용하였다.
    - 기타 Hyper Parameter 및 코드는 추후 Boostcamp가 종료된 후 github을 통해 확인할 수 있다.
- f1 score : 73.277	
- Auprc : 76.317

#### Customized RBERT
![13](https://user-images.githubusercontent.com/53552847/136745007-699b42eb-5338-43a5-815c-3c681a63e8e4.png)
- 구조
    - Backbone model : `klue/roberta-large`
    - typed entity marker : "... [OBJ-PERSON]조지해리슨[/OBJ-PERSON] ... [SUB-PERSON]비틀즈[/SUB-PERSON] ..."
    - RBERT의 경우, 위의 그림에서 볼 수 있듯이 [CLS] token의 사용과 더불어, subject entity token들의 hidden state vector의 평균, object entity token들의 hidden state vector의 평균. 총 3가지를 concatenate하여 이를 토대로 classification을 진행한다.
    - 이에 더해 heuristic한 과정으로서, entity tag token(위의 그림에서의 '$', '#')에 많은 정보가 있을 것이라고 판단하여, 추가적으로 object entity token, subject entity token들의 hidden state vector average를 구하는 과정에서 entity tag token에 대한 정보도 함께 넣어서 연산을 진행하였다.
    - RBERT에 대한 보다 자세한 내용은 [snoop2head log](https://snoop2head.github.io/Relation-Extraction-Code/)에서 확인할 수 있다.
    - 기타 Hyper Parameter 및 코드는 추후 Boostcamp가 종료된 후 github을 통해 확인할 수 있다.
- f1 score : 74.530	
- Auprc : 79.877

#### Customized Baseline Model
- 구조
    - 기존 Baseline(`klue/bert-base`)에서 `klue/roberta-large`로 변경하여 실험을 진행하였다.
    - 기존의 Baseline의 경우, "[CLS] entity1 [SEP] entity 2 [SEP] sentence [SEP] padding"의 형식으로 묶어져 있었는데 이를 Pororo pos tagging을 활용하여 "[CLS] [SUB-TAG]entity1[/SUB-TAG] [SEP] [OBJ-TAG]entity2[/OBJ-TAG] [SEP] sentence [SEP] padding"의 형식으로 수정하여 진행하였다.
    - 위의 두 구조(RE Improved Baseline, RBERT)를 실험하기 전 본 모델을 바탕으로 많은 HyperParameter 실험을 진행하였으며, 일괄적으로 Imbalanced Dataset으로 인한 Focal Loss(gamma = 0.5) 적용과 Stratified KFold를 활용한 Ouf-of-Fold Ensemble을 진행하였다. (추후 실험을 통해 gamma는 1.0으로 수정하여 진행하였다.)
    - 더불어, Unknown Token으로 인하여 `klue/roberta-large` 모델을 적용하였는데, `xlm-roberta-large` 모델과의 성능비교를 통해 본 모델을 최종 선택하게 되었다.
    - 이를 바탕으로 `klue/roberta-large` 모델을 위의 두 모델(RE Improved Baseline, RBERT)의 Backbone Model로 사용하여 진행하였다.
- f1 score : 73.349
- Auprc : 80.332 

#### Ensemble
- 위의 3가지 모델 (Customized Improved Baseline Custom Model, Customized RBERT, Customized Baseline Model)을 최종적으로 Ensemble하여 최종 스코어를 달성하였다.
- Customized RBERT의 경우 f1 score가 가장 높았고, Customized Baseline Model의 경우 auprc score가 가장 높았다. 이에 대하여 Ensemble을 진행하기로 하였다.
- 뿐만 아니라, Ensemble 전략으로서 Customized Improved Baseline Model의 경우 RBERT와 다른 방식으로 진행되었기에 Ensemble 했을 때 성능 향상을 기대할 수 있었기에 총 3가지 모델을 Ensemble하여 최종 스코어를 도출했고 최고의 성능을 낼 수 있었다.

<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
||public|private|
|f1 score|75.218|73.865|
|auprc|81.480|83.216|
{: .tablelines}

## 8. Something Importance
### 8-1. Epoch
- **Customized Baseline Model의 경우, 3epoch** 약 2000 training steps 지점에서 Evaluation Loss가 증가하는 경향을 보였다.
- **Customized RBERT의 경우, 5epoch** 약 3200 training steps까지 Evaluation Loss가 감소하는 양상을 보였다. 이는 Customized Baseline Model에 비해 Classification에서 더 많은 정보를 담고 있기 때문인 것으로 판단된다.

### 8-2. RBERT vs RE Improved Baseline
- **RE Improved Baseline은 모델은 경량화**하며 높은 성능을 유지하는 기법임을 발견할 수 있었다.
    - RBERT의 경우, Out-of-Fold Ensemble을 활용할 때, 5개의 fold를 활용한 결과 총 10h의 학습시간과 26GB의 GPU 용량을 차지하였다.
    - 반면에 RE Improved Baseline은 5개의 fold를 활용하여 학습한 결과 약 5~6h의 학습시간과 15GB의 GPU 용량을 차지하였다.
    - 이에 더하여, 학습 결과 역시 f1 score 0.745(RBERT), 0.736(RE Improved Baseline)로 떨어지긴 했지만 큰 성능차이는 아니므로 RE Improved Baseline이 어느정도 경량화된 모델이라고 판단할 수 있었다.
- RBERT 단점
    - **RBERT의 경우, 각 entity token의 hidden state vector들의 평균을 활용하여 classification을 진행하는데 이 과정에서 각 token들의 순서를 무시**하게된다.
    - 이러한 문제로 인해, 첫 번째 토큰의 hidden state vector만 활용하는 RE Improved Baseline과 큰 성능 차이가 없는 것으로 판단된다.

### 8-3. Loss & Optimizer
#### Loss
- Imbalanced Dataset에 대하여 Weights to CrossEntropy Loss, LabelSmoothing Loss, Focal Loss 적용을 고려하였다.
- 이전 대회의 경험(MASK Competition)을 바탕으로, Focal Loss를 활용 했을 때, Imbalanced Dataset에 대하여 성능 개선이 이루어졌음을 실험적으로 깨달았으며, **본 과정에서 역시 Focal Loss를 활용하여 진행**하였다. (Focal Loss : Cross Entorpy에 Gamma와 Weight를 추가하여 잘 맞추는 Label에 대한 Loss를 줄여 갱신 속도를 늦춰주는 손실함수이다. [출처](https://arxiv.org/pdf/1708.02002.pdf))
- Focal Loss의 gamma를 기존의 Customized Baseline Model에 적용하여 최적값을 찾기 위한 실험을 진행하였으며, 500 step 기준 gamma = 0.5일때 Evaluation Loss - 0.69 / gamma = 1.0일때 Evaluation Loss - 0.56으로 훨씬 좋은 성능을 보였다.
- gamma를 2.0으로 진행하였을 때 마찬가지로 성능이 1.0에 비해 떨어진다는 것을 알 수 있었고 **최종 gamma를 1.0으로 선택하고 RBERT, RE Improved Baseline에 적용**했다.

#### Optimizer
- AdamW와 AdamP 중에서 어떤 optimizer을 사용할 것인지 고민했다.
- 다음의 표에서 볼 수 있듯이 AdamP의 경우 AdamW에 비해 약간 더 빠른 수렴을 하는 것을 알 수 있었고, 저 수치가 큰 차이를 보이는 것은 아니지만 기존의 Vanilla 모델에서의 Adam의 수렴 속도가 23.33이라는 것을 고려해보았을 때, 22.77은 어느 정도 유의미한 성능 향상이라고 판단할 수 있었다. ([AdamP Paper](https://arxiv.org/pdf/2006.08217.pdf)를 참고 하였다.) 
![11](https://user-images.githubusercontent.com/53552847/136739570-122a73fb-c516-4b30-a6bf-b36117a6d7c9.PNG)
- 이러한 정보를 바탕으로 **최종적으로 AdamP optimizer를 활용**하였다.

## 9. Have to Try
- **Uniform Length Batching Strategy**
    - Uniform Length Batching을 활용할 경우, 학습 시간을 유의미하게 줄일 수 있다.
    - HuggingFace의 collate class를 활용하여 적용해보자.
    - collate fn을 활용하여 적용해도록 하자.
- **Data Augmentation**
    - AEDA 적용해보기
    - Round-trip Translation 적용하기. 시간관계 상 약 5000개의 데이터밖에 증가시키지 못했고, 유의미한 성능 개선을 이루지 못했다. 더 많은 데이터 증강을 통해 성능 개선을 유도해보자.
-  **Additional LSTM/GRU Layer**
    - 기존 Backbone Model에 LSTM 혹은 GRU를 한 번 더 거친 후 Classification을 하게 되면 유의미하게 성능 개선이 일어난다는 것이 알려져 있다.
    - 이를 구현하고자 시도하였으나, validation dataset에서 일전 부분이 drop 되는 현상이 발생하였고, torch tensor size에서 miss match가 일어났다.
    - 이러한 이유로 직접 적용하지 못했고, 원인을 찾은 후 적용해보도록 하자.
- **TAPT(Task Adaptive Pretraining)**
    - train/test dataset을 활용하여 backbone model을 본 dataset에 잘 적응할 수 있도록 more pretraining을 진행해보자.
    - 본 과정에서는 기존의 `klue/roberta-large`를 제작할 때, 본 과정의 dataset이 이미 학습됬음으로 파악하여 진행하지 않았지만, 유의미한 성능의 개선이 일어났다고 보고한 다른 팀이 있었다.
    - TAPT는 특정 전문 도메인에서 유의미한 성능 개선이 이루어진다고 널리 알려져 있다.
    - 더불어, TAPT의 경우, 실제 현업에서 보다는 competition에서 보다 자주 사용된다고 한다.
- **HyperParameter Tuning with Optuna**
    - HuggingFace trainer를 활용한 optuna 실험해보기
    - 마지막에 시도를 해보려고 했으나 optuna의 batch_size문제와 학습 시간 상의 문제로 직접 적용하지 못했다.
    - 더불어, 기존의 epoch, batch size 뿐만 아니라 focal loss의 gamma와 같은 hyperparameter들을 적용하는 방법도 찾아보도록 하자.

## 10. 협업
- 1주차
    - Github name branch에서 각자 실험 진행
    - Github Project를 활용한 To do list를 팀원들간 공유
- 2주차
    - dev branch를 개설하여 dev branch에 모두의 코드를 통합
    - 통합한 코드를 바탕으로 실험 진행
    - 최종 코드를 main branch에 업로드

본 과정은 Boostcamp를 진행하며 HappyFace팀으로 참여한 Project이며 이 과정에 대한 Code는 Boostcamp 수료 이후 공개한다.
