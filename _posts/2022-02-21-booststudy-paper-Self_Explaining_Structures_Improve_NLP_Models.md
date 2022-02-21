---
layout: post
title: "[논문리뷰] Self-Explaining Structures Improve NLP Models"
categories: booststudy
tags: paper
comments: true
---
[Self-Explaining Structures Improve NLP Models](https://arxiv.org/pdf/2012.01786.pdf)를 읽고 이에 대하여 논의한다.

## 목차
- [1. Abstract](#1-abstract)
- [2. Introduction](#2-introduction)
- [3. Self-Explaining Framework](#3-self---explaining-framework)
- [4. Experiments](#4-experiments)
- [5. Analysis](#5-analysis)
- [6. Conclusion](#6-conclusion)

## 1. Abstract
- NLP에서의 Deep Learning Model들이 설명가능성에 대하여 겪고 있는 결점이 두 가지 존재한다.
    -  Main Model과 Explaining Model의 결합. => 추가적인 probing model이나 surrogate model이 기존 모델을 해석하기 위해서 사용되어야한다.
    -  설령, Probing Model이 존재한다 하더라도, 개별 단어에 대한 saliency score를 계산하여 low-level feature에서만 작동하여 모델 예측을 설명할 수 있을 뿐이지, high-level text unit에 대해서는 어렵다.
-  위의 두 가지 issue를 다루기 위하여, 간단하지만 일반적이고 효과적인 self-explaining framework에 대하여 제안한다.
-  핵심은 존재하는 **NLP Model의 맨 윗단에 iterpretation layer라고 불리는 additional layer를 넣는 것**이다.
- **Interpretation layer는 특정한 weight와 함께 각각의 text span에 대한 정보를 집계**하여 softmax function에 보내준다.
- 제안되 모델은 다음의 이점이 있다.
    - **span weight을 사용함으로서 self-explainable한 모델을 만들 수 있고 이로서 해석을 위한 추가적인 probing model이 필요없다.**
    - 제안된 모델은 general하며,  존재하는 NLP의 deep learning model들에 잘 적응할 수 있다.
    - **각각의 text span과 관련된 weight는 higher-level text units을 위한 direct importance score를 제공한다.**
- **해석가능성이 성능의 하락으로 이어지지 않는다는 것을 보여준다. 오히려 성능이 상승한다.**

## 2. Introduction
- 오랜기간 deep learning model에 대한 criticism은 해석가능성에 대한 결핍이었다.
- Neural Model의 black-box 특성은 Deep Learning Model의 적용 범위를 상당히 제한할 뿐 아니라 모델의 행동 분석 및 Error 분석을 방해한다.
- **존재하는 해석 모델의 두 가지 결점**
    - **"Not Self-Explainable로 인한 추가적인 Probing Model 요구" + "두 모델의 본질적인 Decoupling으로 인한 Probing Model의 구체적이지 못한 해석 결과"**
    - **Word-level Importance Score에 대한 초점. => Higher-level text unit에 적응하기 어렵다.**
- **무엇이 NLP를 위한 좋은 해석 방법으로 이어질까?**
    - 우선, **self-explainable해야하여, 추가적인 probing model이 필요없어야 한다.**
    - **어떠한 level의 text unit에 대해서도 정확하고 명확한 saliency score를 제공해야한다.**
    - **self-explainalbe하면서 성능과의 tradeoff가 없어야 한다.**
- 위의 3가지 목적에 대하여, NLP context에서의 Deep Neural Model을 위한 self-explainable framework를 제안한다.
- 제안된 framework에서의 key point
    - **interpretation이라고 불리는 additional layer를 기존의 NLP Model의 맨 윗단**에 놓는다,
    - 이 **layer를 바탕으로 모든 text span의 정보를 집계**한다.
    - **각각의 text span은 특정한 weight와 관련되 있으며, weighted combination이 softmax function에 들어가 최종 예측**을 하게 된다.
- 제안된 구조의 이점
    - **self-explainable**하여, 추가적인 probing model이 필요하지 않고 interpretation layer는 objective와 함께 학습된다.
    - **interpretation의 weight를 text span에 대응하는 saliency score로서 바로 사용**할 수 있다.
    - word level을 넘어서 **어떠한 text unit level에서도 직접적으로 정확하고 명확한 saliency score를 제공**한다.
- 일반적인 NLP Deep Learning Structure에 잘 적용할 수 있다.
- 제안된 framework가 정확한 interpretation을 제공하고 error analysis에 용이하다는 것을 보여준다.
- adversarial example generation과 같은 NLP task에 도움을 줄 수 있고, 가장 중요한 것으로는 **처음으로 self-explainable 구조가 성능의 tradeoff를 주지 않고 오히려 더 좋은 성능의 개선**으로 이어졌다는 것이다.

## 3. Self-Explaining Framework
### 3-1. Model
![1](https://user-images.githubusercontent.com/53552847/154935726-bd09dc9c-da05-4506-862b-4aad1091e3bd.png)

#### Input Layer
- 기본적인 NLP의 Deep Learning Model의 input과 유사하다.
- 각 단어들의 representation들이 stack으로 구성되어 있다. D-dimensional vector.

#### Intermediate Layer
- Multi-head Attention, Layer Normalization, Residual Connection으로 구성된 K개의 encoder stack으로 구성되어 있다. (**Backbone Model**)
- position t에 대한, k번째 layer의 representation은 $$h_t^k$$로 나타낸다. 이 때 h는 1xD의 dimension을 가진다.

#### Span Infor Collecting Layer (SIC)
- 임의의 text span의 saliency에 대한 직접적인 측정을 가능하게 하기 위하여, intermediat layer위에 Span Infor Collecting Layer를 놓는다.
- 임의의 text span x(i,j)에 대하여, 우선적으로 x(i,j)에 저장되어야할 모든 semantic하고 syntactic한 정보를 포함하는 h(i,j)를 얻는다. (이때, x(i,j)는 input sequence에서 i번째부터 j번째까지의 span을 의미한다.)
- **구체적으로, x(i,j) span에 대하여 i번째와 j번째. 즉, span에서 맨 처음과 끝의 token에 대한 hidden state vector를 얻고, 이를 F 함수에 넣어서 최종 h(i,j)를 얻게된다.**
    - 이 때, **F는 FFN일 수도 있고, 다른 형태의 Network일 수 있다. **
    - F에 대해서는 자세히 얘기하도록 한다.
- text span을 나타내기 위해서 그 span에서의 starting index와 ending index만을 사용하는 것은 많은 최근 연구에서 사용되었다.
- SIC Layer의 경우 모든 text span에 대해서 참조하기 때문에 시간복잡도 O(N^2)을 따르게 되는데 이를 더 낮추기 위하여 이후에 FFN이 아닌 수정한 형태의 함수를 활용하도록 한다.

#### Interpretation Layer
- **Interpretation Layer에서는 모든 text span에 대한 representation인 h에 대한 모든 정보를 집계한 후 이들을 weighted sum을 활용하여 결합한다.**
- **h를 활용하여, 각 span x(i,j)에 대한 assigning weight alpha(i,j)를 얻는다.**
- alpha(i,j)는 h(i,j)와 매핑되어 scalar로 만들어 준 후에, 모든 alpha에 대해서 normalizing하여 얻어진다.
![2](https://user-images.githubusercontent.com/53552847/154935730-7b7fcea7-6b8c-458c-abad-c11210e90bb6.png)
- $$\hat{h}$$ 는 FFN을 의미하고 h(i,j)는 $$\hat{h}$$ 를 통과하여 scalar가 된다. 이 때, $$\hat{h}$$ 의 dimension은 1xD이다. 
![3](https://user-images.githubusercontent.com/53552847/154935735-63edb36b-b61e-4a80-8622-4736602d7047.png)
- **FFN을 활용하여 alpha(i,j)를 만든 후에, alpha와 h를 summaion하여 최종 representation인 $$\tilde{h}$$ 를 만들어주고 dimension은 1xD이다.**
 
#### Output Layer
![4](https://user-images.githubusercontent.com/53552847/154935737-fb64e45a-156a-4bd4-8562-7216756d8a85.png)
- **기본적인 setup과 유사하게, $$\tilde{h}$$ 를 softmax에 먹여주어 최종 확률값을 출력**하게 된다.
- 이렇게하여, **alpha는 최종 예측을 하는데 있어서 어떤 h(i,j)가 많은 기여를 했는지 알 수 있으며 이를 토대로 x(i,j)의 상대적 중요도를 파악**할 수 있다.
- interpretation layer에서 gradient가 straight-forwardly하게 흐름으로서, text span에 대한 대략적인 값을 계산할 수 있게 된다.

### 3-2. Efficient Computing
- SIC Layer에서 text span에 대한 hidden state vector인 h(i,j)를 구할 때, **만약 F 함수로서 FFN을 사용하게되면, O(D^2)만큼의 계산 복잡도가 사용되며, 이를 모든 text에 대하여 연산한다고 했을 때 O(N^2 * D^2)만큼의 계산 복잡도가 사용**된다.
- 이는 **text가 길어짐에 따라서 감당할 수 없을 정도의 계산량으로 이어지기 때문에, F를 단순히 FFN을 활용하는 것이 아니라 다음의 식을 사용하는 것을 제안**한다.
![5](https://user-images.githubusercontent.com/53552847/154935741-7c549098-b043-4efb-8f22-7fc1722cad43.png)

- 위의 식에서, W는 D to D(dimension이 D차원에서 D차원으로 이동)의 FFN을 나타내며, ⊗는 pairwise dot 연산을 의미한다.
- 이렇게 함으로서 **D^2의 연산 (hidden state vector에서의 모든 i, j의 pair 수)을 단순히 i와 j만 가져와 연산함으로서 D로 줄일 수 있게된다.**
- 위의 과정은 Mou et al(2015), Seo et al(2016)에서 다루고 있으며, 두 semantic representation 사이의 interaction을 모델링 할 때 사용되곤 한다.
 
### 3-3. Training
- training objective는 standard cross entropy loss를 활용한다.
- **추가적인 제약 alpha**가 필요하다.
    - **model은 매우 작은 수의 text span에 초점**을 맞추게 된다.
    - 이러한 원인 때문에, **alpha의 분포는 매우 sharp**하게 된다.
    - 이를 다루기 위하여, 다음의 training objective를 제안한다.
![6](https://user-images.githubusercontent.com/53552847/154935743-048744ae-4379-44f2-8239-823a60202626.png)

- 이 때, 두번째 항은 regularizer로서 사용된다.
- **alpha의 합은 1이라는 제약**이 있으며, **alpha의 제곱합의 경우, alpha가 0과 1로 나뉠경우 높은 값을 가지게 될 것이며, 대부분 같은 값을 가질 경우 작은 값을 가지게 될 것이다.**
- 모델은 end-to-end로 학습된다.

### 3-4. Evaluation
- 모델의 설명가능성과 설명을 믿을 수 있는지를 정확하게 반영할 수 있는지에 대해서 평가하고자 한다.
- 추출된 rationales가 모델 예측과 관련하여 해당 text span이 faithfully represent인지 확인하기 위해서는 다음을 만족해야한다.
    -  기존의 input으로 학습된 모델은 추출된 raitonale로 test 했을 때, 상당히 잘 수행해야 한다.
    -  추출된 rationale로 학습된 모델은 original input으로 test 했을 때, 잘 수행해내야한다.
    -  추출된 rationale로 학습된 모델은 other extracted rationale로 test 했을 때, 잘 수행해내야한다.
-  위의 3가지 측면에서 볼 때, 더 높은 성능은 추출된 rationale이 좀 더 믿을만 하다는 것을 가리키고, 더 좋은 해석 모델이라고 볼 수 있다.
- 이 전략은, 감정분석 task를 위해 서로 다른 감정 측면에 대한 설명으로서 text pieces를 추출하기 위해 rationale generator를 학습시켜, 문장기반 annotation을 기반으로하여 높은 precision을 달성한 Lei et al(2016)에 영감을 받았다.
- 우리는 "full" (original text에 training하고 testing)와 "span" (추출된 rationale에서 모델을 학습하고 testing)을 사용한다.
- 새로운 rationale-based dataset을 구성하기 위하여, rationale-extraction model을 original train dataset에 학습하고 original text로부터 rationale을 추출한다.
- 이후, origianl full input text를 label은 바뀌지 않고 대응하여 추출된 rationale로 대체한다.
- 제안된 평가방법은 완벽하지는 않으며 보안을 위해 다른 evaluation와 결합할 필요가 있다.

## 4. Experiments
### 4-1. Tasks and Datasets
- NLP task (SST-5: text classification, SNLI: natural language inference, IWSLT2014 En to De: machine translation)에서 실험을 진행한다.
- text classification과 NLI에 대해서 backbone model은 RoBERTa를 사용한다.
- alpha를 1/M로 고정한 AvgSelf-Explaining Model을 추가적으로 학습하여 비교한다.
- Machine Translation에서는, Trnasformer-base model을 사용한다.

### 4-2. Main Results
![7](https://user-images.githubusercontent.com/53552847/154935744-7db48f45-897c-4b7d-b619-50610c74242e.png)
- 제안된 Self-Explaining 방법은 SST-5와 SNLI에서 RoBERTa모델을 사용했을 때, 상당한 성능의 개선이 있었다.
- 놀라운 결과로서, **SIC layer와 interpretation layer를 사용한 AvgSelf-Explaining은 여전이 기존 RoBERTa보다 성능이 더 낮다는 것을 발견할 수 있었고, 이는 alpha가 모델 성능에 있어서 중요하다는 것을 발견**했다.

### 4-3. Interpretability Evaluation
- 제안된 Self-Explaining Model을 AvgGradient, AvgAttention, Ratioale Model과 비교한다.
- 모든 방법들은 RoBERTa를 backbone model로 사용하여 비교한다.
![8](https://user-images.githubusercontent.com/53552847/154935747-fad844b6-6d95-4d95-b365-7e09cd67023c.png)

- 모든 setup에서, 우리의 제안된 모델인 Self-Explaining Model의 성능이 가장 좋았다.
- AvgGradient, AvgAttention, Rationale을 비교하였을 때, 각각의 Model이 다른 두 개의 모델보다 일관성있게 더 높고 낮은 결과는 볼 수 없었다. 

### 4-4. The Effect of Lambda
- 추가적인 regularization인 Lambda의 효과를 알아보기 위하여 추가적인 실험을 한다.
- **Lambda가 클수록, alpha의 분포가 더욱 sharp하게끔 강한 영향을 끼치고, 이는 text span에서의 작은 부분을 선호하도록 만든다.**
- 직관적으로 합리적인 Lambda는 모델 성능에 있어서 중요하다.
    - **너무 작은 Lambda는 선택되어진 text span이 모델의 예측을 지지하는데 있어서 확신하지 못한다는 것을 의미한다.**
    - **너무 큰 Lambda는 모델이 예측을 하는데 있어서, 하나의 text span만을 활용했다는 것을 의미한다.**
- 위의 두 가지 경우 중, **전자는 선택된 text-span이 너무 많아지게 만들고, 이는 모델이 예측하는데 있어서 noise를 발생**시킨다.
- 후자는 예측에 있어서 **중요하지 않는 단일 span을 model 예측에 활용하도록 만든다.**
- 그렇기 때문에, 너무 크지도 작지도 않은 적절한 lambda가 필요하다.

![9](https://user-images.githubusercontent.com/53552847/154935749-7a78159d-e243-4f40-92c9-01d2e9025681.png)
- 위의 그림으로부터 볼 수 있듯이, lambda가 1.5일 경우, SST-5에서 가장 좋은 성능을 도출했고, SNLI에서는 lambda가 1.0일 경우에, 가장 좋은 결과가 나왔다.
- lambda가 너무 커졌을 경우에는 성능이 오히려 drastically하게 감소한다.
- 놀라운 사실은, lambda를 주지 않았을 때, 가장 좋았을 때보다는 성능이 더 낮지만, 여전히 baseline보다 성능이 더 좋다.
- 이러한 사실을 바탕으로 제안된 방법이 NLP모델에 있어 성능과 해석가능성 측면 모두에서 개선되었음을 알 수 있다.

## 5. Analysis
### 5-1. Examples
![10](https://user-images.githubusercontent.com/53552847/154935752-c37488b3-b510-4ce4-bd2b-c4ecd344637b.png)
- 위의 표에서 볼 수 있듯이, Self-Explaining을 포함하는 방법론들은 모델이 예측을 하는데 있어서 근거있는 설명으로서 text span을 추출할 수 있다.
- 예를 들어, **Self-Explaining은 "Very Negative"라고 예측하는데 있어서 "overproduced and generally disappointing"을 추출했으며, AvgGradient는 "disappointing"이라는 word를 출력**했다.
- **Self-Explaining, AvgGradient, AvgAttention 모두 올바른 예측을 했음에도 불구하고, AvgGradient와 AvgAttention은 자신들의 예측에 대하여 설명할 수 없다.**
- 대조적으로, Self-Explaining은 positive에서 negative로 연결해주는 접속사 "though"를 성공적으로 포착한다.

### 5-2. Error Analysis
- 잘못된 분류 예제에서 추출된 text span을 조사하여, 모델은 Error Analysis를 수행할 수 있는 직접적인 방법을 제공한다.
- 지난 연구들과 달리 본 모델은 예측 결과로서 word가 아닌 text 단위로 예측하기 때문에 그 오류에 대해서 직접적이고 정확하게 파악할 수 있다.
![11](https://user-images.githubusercontent.com/53552847/154935753-6cd58e20-9535-4fb4-8cd8-7e4bfdb046b9.png)
- 위 Table에서 볼 수 있다시피, 우리는 다음의 몇 가지 실패의 pattern을 식별할 수 있다.
    - **대비 접속사에서 주의해서는 안되는 문장의 일부를 강조한다.**: "[the story is naturally poignant], but first - time screenwriter paul pender overloads it with sugary bits of business" 
    - **모델은 sentiment를 바꾸는 문맥에서 사용되는 단어를 인식할 수 없다.**: "[a well acted and well intention] ed snoozer", "There [isn't a weak or careless performance] amongst them"
    - **irony를 인식할 수 없다.**: "george, [hire a real director] and good writers for the next installment, please"
    - **비유를 인식할 수 없다.**: "It's [like a poem]"

### 5-3. Span-based Adversarial Example Generation
- NLP에서의 Neural Network Model을 공격하기 위한 adversarial examples를 생성하는 것에 흥미가 점점 커지고 있다.
- 현재 adversarial example generation을 위한 current protocoal은 word substitution에 근거하는 것이다. 이는, 중요한 단어를 saliency에 의해 선택하고, 예측을 뒤집을 수 있는 대체제가 있다면 동의로 대체한다.
- 위의 접근 방식에서의 단점은, 단지 단어 수준에서만 수행할 수 있다는 점이다.
    - 이는, saliency score가 word level에서만 신뢰성 있게 계산될 수 있기 때문이다.

![12](https://user-images.githubusercontent.com/53552847/154935755-c7e95b46-d4b3-41b2-bae3-ce55ef330dcc.png)
- 제안된 span-based model은 alpha를 근거로 가장 salient한 span을 식별하고, 대체제가 label을 바꾸게 하는 경우가 있다면, pharaphrase와 함께 대체한다.
- 우리가 아는한, span-based의 접근방식은 NLP에서 adverarial example generationa을 위해 단어 이상의 높은 수준에서 동작하는 첫번째 접근방법이다.
- 위의 방식은 Generating Adversarial Sample을 생성하기 위해 방법론으로서의 Self-Explaining+Paraphrase방법론을 의미한다.
    -  "IMDB", "Yahoo! Answer" dataset을 활용하여 test한다.
    -  다양한 모델들과의 공평한 비교를 위하여 Backbone Model로서 Bi-LSTM을 활용한다.
    -  "Random", "Gradient", "Traversing in Word Order (TiWO)", "Word Saliency (WS)", "Probability Weighted Word Saliency (PWWS)"의 방법론들과 비교한다.
    -  attacking method에 의해 생긴 adversarial sample에 대한 분류 정확도를 위의 표에서 보여준다.
    - 그 결과, Self-Explaining + Paraphrase 방법에 의해 생성된 sample이 사용될 경우, 정확도를 "IMDB", "Yahoo! Answer"에서 각각 84%, 48.86%까지 떨어뜨릴 수 있었다.
    
![13](https://user-images.githubusercontent.com/53552847/154935758-5af4f4f0-b4af-4676-a6f3-857c2e2de1d4.png)

- 위의 표는, adversarial text span을 생성하기 위해, Self-Explaining + Paraphrase를 사용하여 flipping model prediction의 두 가지 예시를 보여준다.
- 위의 예시를 바탕으로, 선택된 text span이 모델 예측에 있어 핵심적이고, span에 주입된 약간의 perturbation으로 인해 예측을 뒤집는다.

## 6. Conclusion
- **NLP contenxt에서 light하지만 효과적인 self-explainable structure에 대하여 제안**한다.
- **interpretation layer를 도입**하고, 각각의 text span에 대한 정보를 집계하고, 이를 바탕으로 model prediction 해석을 위해 기여할 수 있는 specific weight representing을 뽑아낸다.
- 추가적인 실험들을 바탕으로, **성능의 효율 뿐만 아니라 추가적인 모델에 의존하지 않고 self-explaining을 모델에 부여**한다.
