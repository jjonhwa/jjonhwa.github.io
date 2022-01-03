---
layout: post
title: "Vector and Matrix"
categories: booststudy
tags: review
comments: true
---
Vector, Matrix, Lasso and Lidge

## 목록
- [1. Vector](#1-vector)
- [2. Matrix](#2-matrix)
- [3. Lasso and Lidge](#3-lasso-and-ridge)
- [4. ]

## 1. Vector
### 정의
- 벡터는 **숫자를 원소로 가지는 "리스트" 혹은 "배열"**이다.
- 벡터는 공간에서의 한 점을 의미하며 원점으로부터의 상대적 위치를 표현한다.
- 벡터에 스칼라곱을 해 줄 경우, 벡터의 방향은 그대로이고 길이만 변한다.

### 성질
- 벡터끼리의 **차원이 같아야 덧셈 혹은 뺄샘 연산**을 할 수 있다.
    - 차원이 같다 == 개수가 같다.
- Norm: 원점에서부터의 거리
![1](https://user-images.githubusercontent.com/53552847/147902534-9c8870bd-808a-4445-83e7-fc6fbd8e8456.png)
    - **L1 Norm: 절대값의 합**. ex) Robust 학습, Lasso 회귀
    - **L2 Norm: Uclidean Distance**. ex) Laplace 근사, Ridge 회귀
- 각 노름들은 기하학적 성질이 다르다.
![2](https://user-images.githubusercontent.com/53552847/147902536-5ee8ea96-7252-4919-b9e1-49ed41b85997.png)
- 두 벡터 사이의 거리 -> 뺄샘
- 두 벡터 사이의 각도 -> **Cosine Similarity (L2 Norm만 활용가능)**
- 두 벡터를 같은 선(정사영)상에서 x와 y의 곱. ('y선상으로의 x의 정사영'과 'y'의 곱) -> 내적 (두 벡터의 유사도를 측정할 때 사용 가능하다.)

### Code
- Norm
```python
def l1_norm(x):
    x_norm = np.abs(x)
    x_norm = np.sum(x_norm)
    return x_norm

def l2_norm(x):
    x_squared = x ** 2
    x_squared_sum = np.sum(x_squared)
    x_norm = np.sqrt(x_squared_sum)
    return x_norm
```

- 내적 & 각도
```python
def angle(x, y):
    v = np.inner(x, y) # 내적
    cosine = v / (l2_norm(x) * l2_norm(y))
    theta = np.arccos(cosine)
    return theta
```

## 2. Matrix
### 정의
- 벡터를 원소로 가지는 2차원 배열
- 행, 열 index를 가진다.
- 행렬에서 각 행, 열을 행벡터, 열벡터라고 한다.
- 전치행렬: 행과 열의 인덱스가 바뀐 행렬

### 성질
- 각 행렬끼리 차원이 같을 경우 덧셈, 뺄셈 연산을 할 수 있다.
- 성분곱 / 스칼라곱: 각 인덱스 위치끼리의 곱, 각 인덱스에 스칼라를 곱한다.
- 행렬곱: AB라고 할 때, A의 i번째 행벡터와 B의 j번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 계산한다. -> 즉 앞 행렬의 열의 개수와, 뒤 행렬의 행의 개수가 같아아 연산이 가능하다.
- Numpy 내적: i번째 행벡터와 j번째 행벡터 사이의 내적을 성분으로 가지는 행렬
- **모든 선형변환은 행렬곱으로 계산할 수 있다.**

### 역행렬
- 어떤 행렬 A의 연산을 거꾸로 되돌리는 행렬.
- 어떤 행렬 A와 행, 열의 수가 같아야하며 행렬식이 0이 아닌 경우에만 계산할 수 있다.
- 만약 역행렬을 계산할 수 없을 경우에는 Pseudo-inverse 혹은 Moore-Penrose 역행렬을 이용한다.
![3](https://user-images.githubusercontent.com/53552847/147902539-05435ae4-c3ed-430f-9308-13b7aefbf9d4.png)


### code
```python
print(X @ Y) # Numpy에서의 행렬곱 연산
```

```python
x_inv = np.linalg.inv(x) # 역행렬
identity_matrix = x @ x_inv # 기존 행렬 x 역행렬 = 항등행렬
```

```python
x_pinv = np.linalg.pinv(x) # pseudo-inverse
```

```python
# 선형회귀
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_test = model.predict(x_test)

# Moore-Penrose 역행렬
X_ = np.array([np.append(x, [1]) for x in X]) # intercept 항 추가
beta = np.linalg.pinv(X) @ y
y_test = np.append(x_test) @ beta
```

## Lasso and Ridge
- [참고](https://bskyvision.com/193)
- 선형 회귀의 단점을 극복하기 위해 개발된 하나의 방법. 
- 선형회귀의 목적은 실제값과 예측값 사이의 MSE(Mean Squared Error)를 최소화하는 Parameter W(가중치), b(편향)을 찾는 것이다.
- 이런 과정속에서, 주어진 샘플들의 특성과 라벨 사이의 관계를 지나치게 분석함으로서 생겨나는 문제가 Overfitting 문제이고, 간단히 MSE를 최소로하는 W, b만을 찾다보면 이러한 문제가 종종 발생할 수 있다. 이러한 문제를 해결하기 위해 Ridge와 Lasso 방법이 제안되었다.
- 많은 특성들 중 일부만 중요하다면 Lasso가, 중요도가 전체적으로 유사할 경우 Lidge가 더 좋을 수 있다.

### Lasso
- Lasso 식은 다음과 같다.
![4](https://user-images.githubusercontent.com/53552847/147902884-13f2719a-a87c-48f1-834a-5de0403072e3.png)
- MSE가 최소가 되게하는 동시에, 가중치들의 절대값들의 합이 0이 되도록 한다.
- 따라서, 가중치를 0에 가깝게 만드는 제약을 추가함으로서 어떤 특성들은 모델을 만들 때 사용을 제한하고자 하는 것이다.
- 위 식에서 볼 수 있듯이, $$\alpha$$ 값이 중요한데, 이 값이 클 경우 L1-Norm의 값이 커지므로 모델이 학습할 때 L1 Norm을 줄이는 데 더 집중하는 반면, 이 값이 작을 경우에는 MSE를 작아지는데 집중한다고 이해할 수 있다.
- 장점
    - 제약을 추가함으로 인해 일반화된 모델을 더 잘 찾을 수 있다.
    - feature들의 가중치를 0이 되게 함으로써, 모델에서 어떤 특성이 중요한 특성인지 알 수 있도록한다. 즉, 모델 해석력이 좋아진다.
    
### Lidge
- Lidge는 Lasso와 유사하나 제약식을 L2-Norm을 활용한다는 차이점이 있다.
- Lidge는 L2 Norm을 활용하기 때문에, 각 가중치들이 0에 가깝게 만드나 0이 되게 하지는 않는다.
