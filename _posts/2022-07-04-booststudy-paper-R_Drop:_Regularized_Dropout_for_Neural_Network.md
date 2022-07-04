---
layout: post
title: "[논문리뷰] R-Drop: Regularized Dropout for Neural Network"
categories: booststudy
tags: paper
comments: true
---
[R-Drop: Regularized Dropout for Neural Network](https://arxiv.org/pdf/2106.14448.pdf)를 읽고 이에 대하여 논의한다.

## 목차
- [1. Abstract](#1-abstract)
- [2. Method](#2-method)
- [3. Ablation Study](#3-ablation-study)

## 1. Abstract
- Dropout으로부터 생기는 randomness가 Training과 Inference에서의 무시할 수 없는 수준의 불일치의 원인이다.
- **dropout에 의해 생성된 서로 다른 sub model들의 output distribution이 일치하도록 강제**한다. 이 때, 두 모델 사이의 Bidirectional KL-divergence를 최소화함으로서 강제할 수 있다.
- R-Drop은 다양한 task 및 dataset에서 효과적임을 증명하였으며 그 결과는 논문을 참고하도록 하자.

## 2. Method
![image](https://user-images.githubusercontent.com/53552847/177104553-719bd08a-3337-4ad0-aa4c-8faba9863873.png)
- 각 mini-batch 학습에서, **각각의 sampling data는 forward pass를 2번 통과**하게되고, 이는 몇몇 hidden unit들이 random하게 dropout된 서로 다른 model에 의해 처리되어는 것으로 볼 수 있다.
- R-Drop은 **같은 data를 서로 다른 2개의 sub model에 넣고 여기서 출력된 분포를 Bidirectional KL divergence를 최소화함으로서 일치**하도록 만든다.
![image](https://user-images.githubusercontent.com/53552847/177086437-85f0dbc0-058e-42fd-9448-cee12727b531.png)
- NLL Loss를 기본적으로 활용하고, 두 분포를 일치하도록 강제시키기 위한 KL Loss를 추가하여 최종 Loss를 연산한다.
- $$\alpha$$ 는 KL Loss를 control하기 위한 coefficient weight이며, hyperparameter로서 각 task별로 상이하게 변경하여 활용한다.
- **input data x를 `[x; x]` 형태로 concatenate하여 forward pass에 feed**한다.
- 그런 후에, 위의 식의 형태로 Loss를 연산한다.
- 위와 같은 학습 방식은, 각 forward pass step은 'KL divergence'가 없다고 할 때, 2배의 batch size로 학습시키는 시간만큼 소요되는 것과 유사하다.

## 3. Ablation Study
- 매 k step마다 R-Drop을 적용하여 실험을 수행하였지만, `k=1`일 경우에 성능이 가장 좋았다.
- m 개의 Output distribution을 정규화하여 활용한다고 했을 때, `m=3`일 때 BLEU 기준 37.3, `m=2`일 때 BLEU 기준 37.25의 결과가 나왔는데, 이는 m을 2로 잡아도 충분이 정규화의 효과를 볼 수 있다는 것을 알 수 있다.
- Dropout Ratio의 경우, 두 sub model 모두 `0.3`으로 입력했을 때, 가장 성능이 좋았고, reasonable range로서 (0.3 ~ 0.5)를 주었을 때, 큰 변화폭 없이 성능이 좋았다.
- coefficient weight인 $$\alpha$$의 경우 각 task 별로 최적의 값이 다르므로 tuning하여 활용하도록 한다.
