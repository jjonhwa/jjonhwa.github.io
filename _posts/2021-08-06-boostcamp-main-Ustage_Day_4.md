---
layout: post
title: "Ustage Day 4"
categories: boostcamp
tags: main
comments: true
---
Numpy에 대한 내용을 학습하도록 한다.

**부스트 캠프 4일차 학습 요약**
- **행사** : 마스터 클래스, 멘토링
- **학습** : Numpy 수강, 강의 정리 및 선택과제 진행
- **피어 세션**

## 목차
- [1. Numpy](#1-numpy)
- [2. 피어 세션](#2-피어-세션)
- [3. 4일차 후기](#3-4일차-후기)
- [4. 해야할 일](#4-해야할-일)

## 1. Numpy
### 1-1. 어떻게 행렬과 매트릭스를 코드로 표현할 것인가?
- 2차원 List로 표현하는 것이 가장 쉽다.
- 하지만, List로 하면 굉장히 큰 Matrix를 표현하기 힘들다. 더불어 Memory 효율적이지 않다. 더불어, 처리 속도가 느리다.(Interpreter 언어이기 때문이다.)

### 1-2. Numpy란?
- Numerical Python을 의미한다.
- 파이썬의 고성능 과학 계산용 패키지(금융, 증권드에 많이 사용된다.)
- Matrix와 Vector 같은 Array 연산에서의 사실상 표준이다.

### 1-3. Numpy 특징
- 일반적인 List에 비해 빠르고, 메모리 효율적이다.
- 반복문 없이 데이터 배열에 대한 처리 지원(벡터화)
- 선형대수와 관련된 다양한 기능 제공
- C, C++, 포트랑 등의 언어와 통합 가능하다.

### 1-4. ndarray
- 'np.array' 함수를 활용하여 배열을 생성한다.
    - `test_array = np.array([1,4,5,7], float)`
    - `test_array = np.array(['1','4',5.0,8], float)` -> 전부 float로 변환된다.
    - Dynamic Typing을 지원하지 않는다. (List는 지원한다.)
    - Numpy는 하나의 데이터 type만 배열에 넣을 수 있다.
    - C의 array를 사용하여 배열을 생성

### 1-5. Numpy Array의 장점
- Numpy는 바로 값을 메모리에 할당하므로 List는 주소를 할당한다. 그렇기 때문에 List에 비해 Nupy의 효율이 좋다.
- 연산을 할 때, 메모리의 크기가 일정하기 때문에 (각 값에 대한), 데이터를 저장하는 공간을 잡기도 훨씬 효율적이다.
- List는 메모리 비교 연산 'is'를 사용하면 같다고 나오지만 Numpy는 다르다.

### 1-6. 용어
- shape : Numpy Array의 Dimension 구성을 반환
- dtype : Numpy Array의 데이터 type을 반환
- `print(test_array.dtype)`
- `print(test_array.shape)`
![numpy 1번](https://user-images.githubusercontent.com/53552847/128510529-58dfebbe-35c1-43e8-9052-eebe3793036d.PNG)
- shape을 찍게 되면, 3-tensor까지 각각 한 칸씩 밀리게 되서, (4,), (3,4), (4,3,4)가 된다. 각각 (열), (행,열), (채널,행,열
- ndim : number of dimensions -rank를 의미한다.
- size : data의 개수(element의 개수)
- nbytes : numpy array가 차지하는 용량을 확인할 수 있음.

### 1-7. Handling Shape
- reshape : Array의 shape의 크기를 변경함, element의 개수는 동일하다.
    - reshape(-1, 2)를 입력해주게 되면 칼럼을 2로, 행은 자동으로 조정해준다.
- flatten : 다차원 array를 1차원 array로 변환

### 1-8. indexing & slicing
- list와 달리 이차원 배열에서 [0,0] 표기법 제공. (list는 [0][0]으로 표기한다.)
- list와 달리 행과 열을 나눠서 slicing 가능하다.

### 1-9. Creation Function
- Array를 생성하는 함수이다.
- arange 함수
    - array의 범위를 지정하여, 값의 list를 생성하는 명령어
    - `np.arange(0,5,0.5)` : 0부터 5, 전까지 0.5씩으로 생성
- zeros & ones
    - 0, 1로 가득찬 ndarray 생성
    - `np.zeros(shape, dtype, order)`
- empty
    - shape만 주어지고 비어있는 ndarray 생성
    - memory initialization이 되지 않는다.
    - 빈 공간을 잡아주는데 이전에 사용하던 값이 남아있을 수 있다.
    - empty를 사용하는 경우는 많이 없다.
- Something_like(zeros_like, ones_like 등등)
    - 기존 ndarray의 shape 크기 만큼 1, 0 또는 empty array를 반환
    - `test_matrix = np.arange(30).reshape(5,6)`
    - `np.ones_like(test_matrix)`를 하게 되면 5행 6열이 전부 1로 구성된 행렬을 반환한다.
- np.identity(n, dtype)
    - 단위행렬을 입력한다.
- np.eye(row, col ,start_row, dtype)
    - 직사각 대각행렬을 구성
    - start_row는 대각행열을 시작할 열의 위치를 정한다.
- np.diag(matrix, start_index)
    - 대각원소를 추출할 때 사용한다.
    - np.eye와 비슷하게 시작 열(start_index)를 기준으로 대각원소를 추출한다.
    - start_index = -1일 경우 첫 행을 지나서 다음 행의 첫번째 열을 시작으로한 대각원소 추출
- random sampling
    - 데이터 분포에 따른 sampling으로 array를 생성
        - np.random.uniform(시작, 끝, 개수) : 균등분포
        - np.random.normal(평균, 분산, 개수) : 정규분포
        - 등의 다양한 분포를 추출할 수 있다.

### 1-10. Operation Functions
- sum, mean, std등 다음과 같은 식들을 전부 활용할 수 있다.
![numpy 2번](https://user-images.githubusercontent.com/53552847/128510531-b6a00e04-4a35-44c7-a2c8-074c9f87cdd1.PNG)

- axis
    - 모든 Operation Function을 실행할 때 기준이 되는 Dimension 축을 의미한다.
    - Matrix에서 (3,4)라고할 때, 첫 번째 값 3이 axis 0의 값을 의미하고, 4가 axis 1을 의미한다.
    - 축 추가 : b = b[np.newaxis, :]
- concatenate
    - numpy array를 합치는 함수
    - vstack, hstack이 존재한다.
    - concatenate 함수를 활용해 axis를 입력하여 합칠 수 있다.
- Array Operation
    - Numpy는 Array간의 기본적인 사칙 연산을 지원한다.
    - Element Wise operation이다. (shape이 같을 때, 같은 위치의 값들 끼리 연산이 된다.)
- Dot product
    - Matrix의 곱 연산, dot 함수 사용
    - ??@이와 차이가 없는가???
    - A_matrix.dot(B_matrix)
- Transpose
    - 행렬의 전치행렬을 구해주는 함수
    - A_matrix.transpose() 혹은 A_matrix.T 를 활용해 전치한다. 
    - Pytorch, 논문구현에서 많이 사용된다.
- Broadcasting
    - Shape이 다른 배열 간 연산을 지원하는 기능
![numpy 3번](https://user-images.githubusercontent.com/53552847/128517850-a24baeae-bc9f-44b3-8447-850a328bea76.PNG)
- Numpy Performance
    - timeit 매직 메서드를 활용하여 시간을 체크할 수 있다.
    - 일반적인 속도는 ‘for loop < list comprehension < numpy’
    - Numpy는 C로 구현되어 있어, 성능을 확보하는 대신 파이썬의 가장 큰 특징인 Dynamic Typing을 포기했다.
    - 대용량 계산에서 흔히 사용됨.
    - Concatenate와 같은 할당을 하는 연산에서는 연산 속도의 이점이 없다. 계산에서의 연손 속도 이점이 있다.

## 2. 피어 세션
- [이전 질문 한번 더]
  - ReLU가 왜 비선형인가
  - ReLU와 Sigmoid 함수의 차이
  - KL-Divergence에 대한 내용 정리 [여기](https://angeloyeo.github.io/2020/10/27/KL_divergence.html)
  - Entropy, Cross Entropy(예측과 달라서 생기는 정보량)
 
- [금일 질문 목록]
  - RNN에서 시퀀스가 너무 길면 Vanishing, Exploding이 왜 발생하는가?
  - 퀴즈 -> 베이즈 정리에 대한 문답
  - 흔히 말하는 Gradient란? -> 변수 축에 따른 Gradient라고 생각하면 이해하기 쉽다. (x축 Gradient는 x축에 따라서 미분값을 활용한 하강 혹은 상승)
  - Mini-Batch에서의 복원추출과 비복원추출
  - RNN에서의 Gradient 방법은?
  - 미분의 근사에서 도함수를 통해 컴퓨터에서 정확히 사용할 수 있는가? ($$x^2$$의 도함수는 2x인데 미분근사 식을 이용안하고 바로 2x에 대입해서 써도 되는가?)

## 3. 4일차 후기
강의를 정리하는 데 정말 시간이 많이 걸린다는 것을 알았고, 이러한 학습으로부터의 능률 역시 이전의 공부에 비해서 훨씬 높다는 걸 체감하고 있다.
더불어, 강의 중간중간 그리고 팀원들과의 소통을 통해서 몰랐던 내용들을 개인 학습을 통해서 많이 진행하게 되면서 공부 방법 및 많은 지식들의 습득을 체감하고 있는 중이다.
**물론 그에 다른 나의 몸은 녹초가 되어가고 있다..ㅜㅜ**

오늘은 이전 강의들에 대한 정리와 과제를 진행했는데 어제 느꼈던 것처럼 과제는 정말 계속 어려워지는 것 같다..
그래도 해냈다는 것 하나에 의미를 두면서 열심히 해보고자 한다!

## 4. 해야할 일
- ndim이 rank를 의미하는가??
- garbage collection
- @와 dot product의 차이
