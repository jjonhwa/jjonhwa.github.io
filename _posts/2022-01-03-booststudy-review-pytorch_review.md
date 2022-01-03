---
layout: post
title: "PyTorch"
categories: booststudy
tags: review
comments: true
---
PyTorch, Tensorflow, Tensor, view and reshape

## 목록
- [1. PyTorch vs Tensorflow](#1-pytorch-vs-tensorflow)
- [2. Tensor](#2-tensor)
- [3. 추가공부](#3-추가공부)

## 1. PyTorch vs Tensorflow
### PyTorch
- Define by Run: 실행을 하면서 그래프를 생성하는 방식
    - 즉시 확인가능하며, pythonic code
- GPU support, Good API and community
- 사용하기 편한 장점
- 자동미분을 지원하여 DL 연산을 지원
- 다양한 형태의 DL을 지원하는 함수와 모델을 지원함.

### Tensorflow
- Define and Run: 실행시점에 데이터 feed
- production과 scalability의 장점

## 2. Tensor
### Tensor란?
- 다차원 Arrays를 표현하는 PyTorch 클래스
- 사실상 numpy의 ndarray와 동일. (Tensorflow의 Tensor와 동일)
- Tensor를 생성하는 함수 역시 거의 동일하다.
- 기본적으로 tensor가 가질 수 있는 data 타입은 numpy와 동일하다.
- PyTorch의 tensor는 GPU에 올려서 사용가능

### Tensor handling
- view, squeeze, unsqueeze 등으로 tensor 조정가능
- view: reshape과 동일하게 tensor의 shape을 변환
- squeeze: 차원의 개수가 1인 차원을 삭제 (압축)
- unsqueeze: 차원의 개수가 1인 차원을 추가
- view vs reshape : contiguity 보장의 차이
- 행렬곱셈 연산의 경우 mm을 활용 (tensor에서는 dot을 활용하지 않는다.)
- mm과 matmul은 broadcasting 지원 차이
- nn.functional 모듈을 통해 다양한 수식 변환을 지원한다.
- backward 함수를 통한 자동 미분을 지원한다.

### code
```python
# tensor 생성
import torch
n_array = np.arange(10).reshape(2, 5)
t_array = torch.FloatTensor(n_array)
print(t_array.ndim, t_array.shape)
```

```python
# list to tensor / numpy to tensor
data = [[3, 5], [10, 5]]
x_data = torch.tensor(data)

nd_array = np.array(data)
tensor_array = torch.from_numpy(nd_array)
```

## 추가공부
### view vs reshape
- view와 reshape의 차이는 contiguity 차이이다.
- Contiguous: Torch Array의 값이 연속적인 메모리 위치를 가지는 상태
- view: contiguous tensor에서만 작동함. -> contiguous tensor를 반환
- reshape: contiguous tensor에서는 view와 동일하게 작동. non-contiguous tensor에서는 data를 copy한다.
- 아래 코드를 실행
```python 
a = torch.zeros(3, 2)
b = a.t().reshape(6)
a.fill_(1)

print(a)
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
print(b)
tensor([0., 0., 0., 0., 0., 0.])
```

```python
a = torch.zeros(3, 2)
b = a.view(6)
a.fill_(1)

print(a)
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
print(b)
tensor([1., 1., 1., 1., 1., 1.])
```

## 해야할 것
- view와 reshape은 contiguity 보장의 차이란 무엇인가?
- 
