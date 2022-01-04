---
layout: post
title: "Gradient Descent"
categories: booststudy
tags: review
comments: true
---
Gradient Descent (경사하강법)

## 목록
- [1. Gradient Descent Basic](#1-gradient-descent-basic)
- [2. Gradient Descent Advanced](#2-gradient-descent-advanced)

## 1. Gradient Descent Basic
### 미분
- 미분을 활용하여 함수의 **기울기를 찾을 수 있다.**
- 기울기를 바탕으로 어느 방향으로 움직여야 함수값이 증가 혹은 감소하는 지 알 수 있으며, 이를 바탕으로 경사하강 혹은 경사상승이 진행된다.
- 2차함수를 예시로 생각할 때, 미분값을 빼면 경사하강이, 미분값을 더하면 경사상승이 이루어지며, 흔히 **경사하강법이라고 불리는 방법론은 기존값에서 미분값을 뺀 값으로 update하는 방식으로 진행된다.**
![5](https://user-images.githubusercontent.com/53552847/148011557-81272f04-68e6-4d1b-bd64-dfac844f1ea2.png)
![6](https://user-images.githubusercontent.com/53552847/148011558-27341c25-d2d6-4e5d-a9f6-c223eab4b749.png)
- 미분값은 극점에서 0에 가까워지므로, 경사하강 혹은 경사상승은 극점에 도달할 경우 더 이상 움직이지 않게 된다. 이런 방법론을 바탕으로 모델의 최적화가 진행된다.

### 경사하강법
#### 변수가 scalar일 경우

```python
def func(val):
    '''함수정의'''
    fun = sym.poly(x**2 + 2*x + 3)
    return fun.subs(x, val), fun

def func_gradient(fun, val):
    '''미분'''
    _, function = fun(val)
    diff = sym.diff(function, x) # 함수를 x에 대하여 미분
    return diff.subs(x, val), diff
    
def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
    """경사하강법"""
    cnt = 0 # 횟수
    val = init_point # 초기값
    diff, _ = func_gradient(fun, init_point)
    while np.abs(diff) > epsilon:
        val = val - lr_rate*diff  # 값 update
        diff, _ = func_gradient(fun, val) # 경사하강 진행
        cnt += 1
        
    print(f"함수: {fun(val)[1]}, 연산횟수: {cnt}, 최소점: ({val}, {fun(val)[0]})")
    
gradient_descent(fun=func, init_point=np.random.uniform(-2,2))
```

#### 변수가 벡터일 경우
- 벡터가 입력일 경우, 편미분을 활용한다.
- 각 변수 별로 편미분을 계산한 Gradient Vector를 이용하여 이를 경사하강/경사상승에 사용할 수 있다.
![7](https://user-images.githubusercontent.com/53552847/148011559-5b4b8fe3-9025-4875-91f2-8d01c7d3c421.png)

```python
def eval_(fun, val):
    val_x, val_y = val
    fun_eval = fun.subs(x, val_x).subs(y, val_y)
    return fun_eval
        
def func_multi(val):
    x_, y_ = val
    func = sym.poly(x**2 + 2*y**2)
    return eval_(func, [x_, y_]), func
    
def func_gradient(fun, val):
    x_, y_ = val
    _, function = fun(val)
    diff_x = sym.diff(function, x)
    diff_y = sym.diff(function, y)
    grad_vec = np.array([eval_(diff_x, [x_, y_]), eval_(diff_y, [x_, y_])], dtype=float)
    return grad_vec, [diff_x, diff_y]
    
def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
    cnt = 0
    val = init_point
    diff, _ = func_gradient(fun, val)
    while np.linalg.norm(diff) > epsilon:
        val = val - lr_rate*diff
        diff, _ = func_gradient(fun, val)
        cnt += 1
    print(f"함수: {fun(val)[1]}, 연산횟수: {cnt}, 최소점: ({val}, {fun(val)[0]})")

pt = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
gradient_descent(fun=func_multi, init_point=pt)
```

## 2. Gradient Descent 2단계
### 경사하강법 응용
- 선형회귀식을 구할 때, 역행렬 대신 경사하강법을 이용해 찾을 수 있다.
![8](https://user-images.githubusercontent.com/53552847/148011545-c5b7c094-2a2c-432b-8dc0-b228a5aa2b4c.png)
- 기존 선형회귀에서 역행렬을 구할 때, 위의 그림과 같이 beta를 구한 후, 새로운 x와 베타를 연산하여 예측값을 구하게 된다.
![9](https://user-images.githubusercontent.com/53552847/148011547-e9ca8bb8-f722-4bf7-abbc-13dd878513b8.png)
- 이러한 과정에서 역행렬을 구하는 대신 위의 그림과 같이 선형회귀의 목적식을 최소화하는 beta를 찾는 방법으로 Gradient Vector를 구하여 예측값을 찾을 수 있다.
![10](https://user-images.githubusercontent.com/53552847/148011549-82544044-076c-4359-b134-41d41a828242.png)
![11](https://user-images.githubusercontent.com/53552847/148011550-9971db36-2a7d-48a2-81e2-67591e004404.png)
- 위의 식을 간단히 정리하면 다음과 같고, beta를 업데이트 하는 방식으로 최적의 해를 찾는 방향으로 진행한다.
- 경사하강법에서는 위의 식에서도 알 수 있다시피, 학습률이 중요한 hyperparameter이고 학습횟수 역시 어느정도 보장이 되어야한다.

```python
X = np.array([[1,1], [1,2], [2,2], [2,3]])
y = np.dot(X, np.array([1,2]) + 3

beta_gd = [10.1, 15.1, -6.5] # beta의 초기값. 목표값 [1,2,3]
X_ = np.array([np.append(x, [1]) for x in X]) # intercept 항 추가
lr = 0.01

for t in range(5000): # 학습횟수
    error = y - X_@beta_gd
    grad = - np.transpose(X_) @ error
    beta_gd = beta_gd - lr*grad # beta update
print(beta_gd)
```

### 왜 경사하강법을 사용할까?
- 경사하강법은 미분가능하고 볼록한 함수에서는 학습률과 학습횟루를 적절히 선택했을 때, 수렴이 보장된다.
- 하지만, 비선형회귀문제의 경우, 목적식이 볼록하지 않을 수 있고 수렴을 항상 보장하지는 않는다.
- 확률적 경사하강법 (Stochastic Gradient Descent, 이하 SGD)는 모든 데이터를 사용하지 않고 데이터의 일부만 활용하여 Gradient를 업데이트한다.
- non-convex에 대하여 SGD를 통해 최적화를 진행할 수 있고, SGD 역시 항상 잘 작동하는 것은 아니지만 기존의 Gradient Descent (모든 데이터 활용) 보다는 실증적으로 더 낫다고 검증되었다.
- SGD의 경우, 데이터의 일부를 활용하기 때문에, 연산자원을 좀 더 효율적으로 사용할 수 있고 식은 다음과 같다.
![12](https://user-images.githubusercontent.com/53552847/148011551-31a7c3dc-d1d6-4c0c-971f-110db00b9057.png)
- SGD의 원리는 다음과 같고, Mini Batch Dataset을 활용하여 새로운 그래프를 그리고 각 점에서의 기울기값을 활용해 경사하강이 진행된다.
![13](https://user-images.githubusercontent.com/53552847/148011555-c0e4b2ec-242a-4e22-9a6c-3db5d1ebdc9c.png)
