---
layout: post
title: "Singular_Value_Decomposition(특이값_분해)"
categories: booststudy
tags: plus
comments: true
---
특이값 분해(Singular Value Decomposition)에 대한 기본적인 내용, 기하학적 의미, Pseudo Inverse에 대한 내용을 다룬다.

본 내용은 https://darkpgmr.tistory.com/106 를 참고하여 작성하였다.

## 특이값 분해 (Singular Value Decomposition)

### 1. 용어 및 기본 내용
- 특이값 분해는 행렬을 대각화하는 한 방법을 의미한다.
- 행렬이 정방행렬이든 아니든 적용 가능하기 때문에 유용하다.(고유값 분해(Eigen Decomposition의 경우 정방행렬일 경우에만 적용 가능하다.)

**NOTE : 고유값 분해에 대한 내용은 [여기](https://darkpgmr.tistory.com/105)에서 확인하도록 하자.**(Eigen Decomposition)

우리는 A를 U$$\sum$$$$V^T$$로 나타낼 수 있고 이렇게 쪼개는 것을 특이값 분해(Signular Value Decomposition)이라고 한다.
- U는 $$A$$$$A^T$$를 고유값 분해(Eigen Decomposition)을 통해 얻은 직교행렬이며, left singular vector라고 한다.
- V는 $$A^T$$$$A$$를 고유값 분해(Eigen Decomposition)을 통해 얻은 직교행렬이며, right signular vector라고 한다.
- $$\sum$$은 U와 V에서 각각 얻은 Eigen Value의 제곱근을 원소로 가지는 직사각 대각행렬이며, 이 때 대각에 존재하는 원소를 singular value이라고 한다. Singular Value 값들은 전부 0 또는 양수의 값만을 가진다.
    - 직사각 대각행렬은 다음 그림과 같은 행렬을 의미한다.
    ![1번](https://user-images.githubusercontent.com/53552847/128434934-0905aa17-4909-4f95-9dc0-3855450ccd1d.png)

- 직교행렬의 특징으로는 $$U$$$$U^T$$ = $$V$$$$V^T$$ = I, $$U^T$$ = $$U^{-1}$$, $$V^T$$ = $$V^{-1}$$ 이다.

### 2. SVD의 기하학적 의미

 `x' = Ax`라는 식이 있을 때, 행렬 A는 좌표공간에서 선형변환으로 이해할 수 있다. (행렬을 이해하는 방법 중 하나)
 
 이 때, 선형변환에 있어 각 행렬은 다음과 같은 의미를 지닌다.
 - 직교행렬의 기하학적 의미는 회전변환 혹은 반전된 회전변환을 의미한다.
 - 대각행렬의 기하학적 의미는 각 좌표 성분으로의 스케일 변환을 의미한다.
 - 더불어, 직교행렬의 행렬식은 항상 1 or -1을 가지게 되는데, 1일 경우에는 일반적인 회전변환을, -1일 경우에는 반전된 회전변환을 의미한다.

**NOTE : 행렬식에 대한 내용은 [여기](https://ko.wikipedia.org/wiki/%ED%96%89%EB%A0%AC%EC%8B%9D)에서 확인하도록 하자.** (Determinant)

따라서, A = $$U$$$$\sum$$$$V^T$$로 나타냈을 때, U, V는 직교행렬, $$\sum$$은 직사각 대각행렬이므로 Ax는 $$U$$$$\sum$$$$V^T$$x로 나타낼 수 있고 순서와 그림은 다음과 같다.
- $$V^T$$x : x의 회전변환
- $$\sum$$ x' : x'의 스케일변환
- Ux'' : x''의 회전변환
![2번](https://user-images.githubusercontent.com/53552847/128434955-13a2f138-32d8-4a02-bbed-0e447638aab2.png)

- 더불어, Singular Vector(U, V)는 회전변환을 시키고, Singular Value($$\sum$$)은 스케일 변환을 시킨다는 것을 알 수 있다. 즉, 도형의 형태적 변화는 오직 A의 Singular Value에 의해서만 결정됨을 알 수 있다.

### 3. Reduced SVD와 행렬 근사를 통한 데이터 압축
A가 $$m$$ x $$n$$ 행렬일 떄, SVD를 활용하는 분해 그림은 다음과 같다.
![3번](https://user-images.githubusercontent.com/53552847/128434959-54c98d2c-8390-446f-9668-3b4a090549ea.png)

위는 Full SVD를 의미하는데, 보통 FUll을 사용하는 경우는 드물고 보통 Reduced SVD를 사용한다. 왜 Reduced를 사용하는지 다음으로부터 알아보자.

1. thin SVD :
![4번](https://user-images.githubusercontent.com/53552847/128434971-5bacb55b-eb81-4937-a451-853626049b8b.png)
U를 s만큼만 활용하는 것을 볼 수 있는데, $$\sum$$에서 Singular Value를 포함하는 대각행렬의 크기를 제외하고는 전부 0행렬이므로 대각행렬만 사용하여도 결국 결과값은 A로 같다는 것을 알 수 있다.

2. Compact SVD :
![5번](https://user-images.githubusercontent.com/53552847/128434984-76380083-c589-4c85-8c89-d718cfa6404f.png)
r은 Singular Value에서 0인 값들을 전부 제거하고 0이 아닌 대각 원소만 남겨놓은 형태이다. 이 경우에 역시 0이 아닌 특이값만 남겨놈으로서 계산을 해주게 되면 결과가 A로 동일하게 된다.

3. Truncated SVD :
![6번](https://user-images.githubusercontent.com/53552847/128435000-484dfe93-d4dd-4042-8ef0-c754007db719.png)
Truncated SVD의 경우 Singular Value가 0이 아닌 값들 역시 제거함으로서 A가 그대로 보존되지 않고 A에 근사하는 A'가 된다.

- 여기서 흥미로운 점은, 이렇게 근사하는 A'가 Matrix Norm || A - A' ||를 최소화 하는 Rank t 행렬로서 데이터 압축, 노이즈 제거에 활용될 수 있다는 점이다.
** Note : Rnak에 대한 내용은 [여기](https://gosamy.tistory.com/16)을 참고하자** 

즉, 다음의 링크에서 볼 수 있듯이 t를 줄이면 줄일수록 이미지가 압축되는 것을 볼 수 있다.
https://darkpgmr.tistory.com/106

- 좋은 압축 방법은 아니지만 Truncated SVD를 통한 데이터 근사가 원래의 데이터 핵심을 잘 잡아내고 있음을 알 수 있다.

### 4. SVD와 Pseudo Inverse
#### 1. pseudo inverse란?
- 'Ax = b'에서 A가 정방행렬이 아닐 경우 A의 역행렬 대신 사용하는 $$A^+$$
- 위 식에서 해는 x = $$A^+$$ b가 되고 이 때, x는 \| Ax - b \|(Loss)를 최소화하는 해가 된다.

#### 2. Pseudo Inverse와 SVD는 어떤 관게?
- SVD는 A를 $$U$$$$\sum$$$$V^T$$로 쪼갤 수 있다는 것을 알 수 있다. 이 때, pseudo inverse인 $$A^+$$ 는  $$V$$$$\sum^+$$$$U^T$$로 계산된다.
- 앞서 설명한 것으로부터, U, V는 직교행렬로서 $$U^T$$ = $$U^{-1}$$, $$V^T$$ = $$V^{-1}$$ 임을 알 수 있고 $$\sum$$만이 $$\sum^+$$ 로 변환된 것을 알 수 있다. $$\sum^+$$ 의 경우 다음 그림을 보고 이해할 수 있다.
![7번](https://user-images.githubusercontent.com/53552847/128435014-08f6c9e0-183f-4b24-93ff-69b795367b43.png)

#### 3.pseudo inverse $$A^+$$ 의 의미
- $$A^+$$ 의 의미 역시 Singular Value에 의해 결정된다.
- Singular Value는 $$\sum$$의 대각원소 부분을 의미한다는 것을 우리는 알고 있다. 이 때, 모든 Singular Value가 양수일 경우 우리는 다음 식으로부터 $$A^+$$ $$A$$, $$A$$ $$A^+$$ 가 단위행렬이 됨을 알 수 있다.
![8번](https://user-images.githubusercontent.com/53552847/128435024-64a4fd63-5283-4ad3-8bf0-8b5973fcfdd5.png)

- 더불어, Singular Value의 값이 모두 양수가 아닌 0을 포함하고 있다면 앞서 설명한 Truncated SVD와 비슷한 방식으로 해석할 수 있는데, 0인 부분을 제거하여 데이터를 압축한다. 이로 부터 $$A^+$$ 는 A로 근사함을 어느 정도 이해할 수 있다.
- 또한, Singular Value의 양수값이 0에 가까울 경우에 노이즈로 판단하여 0으로 치환한 후 pseudo inverse를 구하는 것이 일반적이라고 한다. 이 역시, Truncated SVD를 통해 어느정도 이해할 수 있다.(근사 방법)

- 마지막으로, 앞선 수업시간(BoostCamp)에서 선형연립방정식의 경우 m > n 인 행렬일 경우에만 대상으로 pseudo inverse를 활용하여 근사해를 구할 수 있다고 하였고, 이는, 식의 개수가 변수의 개수보다 많다는 것을 뜻한다고 하였다. 이 떄, pseudo inverse의 입장에서 이해하자면, m > n 일 경우에 $$A^+$$ $$A$$ 가 단위행렬로 근사하고, n > m 일 경우에는 $$A$$ $$A^+$$ 가 단위행렬로 근사한다는 것을 알고 있으므로, Ax = b에서 $$A^+$$ $$A$$의 형태로 곱해줘야지 x = $$A^+$$ b의 결과가 나와 근사해를 구할 수 있음을 알 수 있다.

Colab Markdown :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZsGj99fgw1Op1M83I-gRAbpTwMjqY9Vk?usp=sharing)
