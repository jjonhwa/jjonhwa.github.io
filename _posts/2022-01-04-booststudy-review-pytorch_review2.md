---
layout: post
title: "PyTorch Module and Optimize"
categories: booststudy
tags: review
comments: true
---
PyTorch Module 및 최적화 과정 설명

## 목록
- [1. PyTorch Module](#1-pytorch-module)
- [2. PyTorch Optimize](#2-pytorch-optimize)

## 1. PyTorch Module
### python notebook (.ipynb) vs python (.py)
- 초기 단계
    - 대화식 개발 과정인 notebook 형태가 유리하다.
    - 학습 과정 및 디버깅을 지속적으로 확인하며 진행할 수 있다.
- 초기 이후
    - 배포 및 공유에서는 notebook 공유의 어려움이 있다.
    - 재현의 어려움이 따르며, 실행 순서 역시 꼬일 수 있다.
    - python file을 활용하여 개발의 용이성을 확보할 수 있고 유지보수를 더욱 쉽게 진행할 수 있다.

### Module 구성
![14](https://user-images.githubusercontent.com/53552847/148016333-2cb149da-fb7e-4891-ac7b-6a24a23c6b6d.png)

## 2. PyTorch Optimize
### torch.nn.Module
- 딥러닝을 구성하는 Layer의 base class
- Input, Output, Forward, Backward 정의
- 학습의 대상이 되는 parameter(tensor) 정의

### nn.Parameter
- Tensor 객체의 상속 객체
- nn.Module 내에 attribute가 될 때는 `required_grad=True`로 지정되어 학습 대상이 되는 Tensor로 지정
- 우리가 직접 지정할 일은 잘 없으며, 대부분의 layer에는 weights 값들이 지정되어 있다.

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.randn(in_features, out_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x: Tensor):
        return x @ self.weights + self.bias
```

### Backward
- Layer에 있는 Parameter 들의 미분을 수행
- Forward의 결과값 (model의 output)과 실제값 사이의 차이(Loss)에 대하여 미분을 수행
- 해당 값으로 Parameter Update

```python
for epoch in range(epochs):
    optimizer.zero_grad() # optimizer 초기화
    outputs = model(inputs) # output 출력
    loss = criterion(outputs, labels) # 예측값과 실제값 사이의 loss 계산
    loss.backward() # backward로 부터 미분
    optimizer.step() # 값 update
```

- 실제 backward는 Module 단계에서 직접 지정이 가능하다.
- Module에서 backward와 optimizer 오버라이딩
- 사용자가 직접 미분 수식을 써야하기 때문에, 잘 쓸일은 없지만 순서는 이해할 필요가 있다.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class LR(nn.Module):
    def __init__(self, dim, lr=torch.scalar_tensor(0.01)):
        super(LR, self).__init__()
        
        self.w = torch.zeros(dim, 1, dtype=torch.float).to(device)
        self.b = torch.scalar_tensor(0).to(device)
        self.grads = {
            "dw": torch.zeros(dim, 1, dtype=torch.float).to(device)
            "db": torch.scalar_tensor(0).to(device)
        }
        self.lr = lr.to(device)
        
    def forward(self, x):
        z = torch.mm(self.w.T, x)
        a = self.sigmoid(z) # activate
        return a
        
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))
        
    def backward(self, x, yhat, y):
        self.grads['dw'] = (1/x.shape[1]) * torch.mm(x, (yhat - y).T)
        self.grads['db'] = (1/x.shape[1)) * torch.sum(yhat - y)
        
    def optimize(self):
        self.w = self.w - self.lr * self.grads['dw']
        self.b = self.b - self.lr * self.grads['db']
```

- 위의 코드에서 backward와 optimize의 경우 아래 식의 방식으로 진행된다.
![15](https://user-images.githubusercontent.com/53552847/148016454-258d3dca-433a-4f8e-bcc9-0f180c99274f.png)
![16](https://user-images.githubusercontent.com/53552847/148016457-2292bd8f-f66f-4fa0-9d1e-24fadf67bcf8.png)



