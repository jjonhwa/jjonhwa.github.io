---
layout: post
title: "Code Explanation(Drawing_Using_OpenCV)"
categories: dev
tags: opencv
comments: true
---

본 장은 [Drawing Using OpenCV](https://github.com/jjonhwa/Drawing_Using_OpenCV)에 대한 **OpenCv 코드 설명 페이지**이며 본 장에서 설명되어진 코드는 [Code Review(Drawing Using OpenCV)](https://jjonhwa.github.io/2021-06-07-Code_Review/)에서 활용합니다.

### Notification
{: .box-note}
**Note:** 다음의 코드 설명에서는 필수 입력 값들만 입력했으며 보다 많은 옵션들이 존재하는 코드들이 있습니다.

## 목차

[1. cv2.imread](#1-cv2imread)  
[2. cv2.cvtColor](#2-cv2cvtcolor)  
[3. cv2.threshold](#3-cv2threshold)  
[4. cv2.Canny](#4-cv2canny)  
[5. 모폴로지 연산 (cv2.getStructuringElement, cv2.dilate, cv2.erode, cv2.mophologyEx)](#5-모폴로지-연산-cv2getstructuringelement-cv2dilate-cv2erode-cv2mophologyex)  
[6. cv2.getStructuringElement - 모폴로지 연산](#6-cv2getstructuringelement---모폴로지-연산)  
[7. cv2.dilate - 모폴로지 연산](#7-cv2dilate---모폴로지-연산)  
[8. cv2.erode - 모폴로지 연산](#8-cv2erode---모폴로지-연산)  
[9. cv2.morphologyEx - 모폴로지 연산](#9-cv2morphologyex---모폴로지-연산)  
[10. cv2.findContours](#10-cv2findcontours)  
[11. cv2.drawContours](#11-cv2drawcontours)  
[12. cv2.boundingRect](#12-cv2boundingrect)  
[13. cv2.contourArea](#13-cv2contourarea)  
[14. cv2.line](#14-cv2line)  
[15. cv2.GaussianBlur](#15-cv2gaussianblur)  
[참고 URL](#참고-url)

## 1. cv2.imread

imread는 **이미지 파일을 읽을 때 사용**한다.

```
img = cv2.imread(image_path, flag) >> 코드로 출력
```

image_path : 이미지 파일의 경로를 입력해준다.

flag : 이미지 파일을 읽을 때 옵션이다. 옵션의 종류는 다음과 같다.
  - cv2.IMREAD_COLOR (1) : 이미지 파일을 Color로 읽는다. 투명한 부분은 무시하며 Default이다.
  - cv2.IMREAD_GRAYSCALE (0) : 이미지 파일을 GrayScale로 읽는다.
  - cv2.IMREAD_UNCHANGED (-1) : 이미지 파일을 alpha channel까지 포함해서 읽는다.



```python
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
print(img)
cv2_imshow(img)
```

    [[[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     ...
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]]



![1번](https://user-images.githubusercontent.com/53552847/120942685-8f049a80-c765-11eb-904e-f446b0766d0b.png)


## 2. cv2.cvtColor



cvtColor는 Convert Color를 의미하며 색상 공간 변환을 뜻한다. 이 함수는 **본래의 색상 공간에서 다른 색상 공간으로 변환할 때 사용**한다.

- cvtColor 함수는 데이터 타입을 같게 유지하고 채널을 변환한다.
- 입력 이미지는 8, 16, 32 비트의 정밀도를 갖는 배열을 사용할 수 있다.
- 출력 이미지는 입력 이미지의 크기와 정밀도가 동일한 배열이다.
- 채널의 수가 감소되어 이미지 내부의 데이터는 설정한 색상 공간과 일치하는 값으로 변환되며, 데이터 값이 변경되거나 채널 순서가 변경될 수 있다.


```
gray = cv2.cvtColor(img, code)
```

img : 입력 이미지(numpy array)

code : 색상 변환 코드로 `원본이미지색상공간2결과이미지색상공간`으로 표시한다.
- 색상 공간 코드는 흔히 COLOR_BGR2GRAY를 많이 사용한다.
- 더불어 HSV, Luv 등의 색상코드가 있으며 다양한 색상코드에 대한 설멸은 [Python OpenCV 강좌 : 제 10강 - 색상 공간 변환](https://076923.github.io/posts/Python-opencv-10/)에서 찾아보도록 하자.


```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray) # 기존의 이미지가 흑백처리되어있으 다음 출력이 동일하다.
```


![2번](https://user-images.githubusercontent.com/53552847/120942687-9035c780-c765-11eb-8776-4d5f58a1461b.png)

## 3. cv2.threshold




이미지에서 이미지 픽셀갑이 thrshold값보다 크면 입력된 고정 값으로 할당하고 작으면 입력된 다른 고정 값으로 할당한다.

**cv2.threshold를 적용하기 위해서는 grayscle로 변환한 후에 적용해야 한다.**


```
ret, thresh = cv2.threshold(gray, threshold_value, value, flag)
```
gray : Grayscale된 이미지

threshold_value : 픽셀 경계값

value : 픽셀 경계값보다 클 경우 적용되는 최대값. (적용되는 플래그에 따라서 픽셀 경계값보다 작은 경우 적용되는 최대값을 의미하기도 함.)

flag : 경계값 적용 방법 또는 스타일
  - cv2.THRESH_BINARY : 픽셀 값이 경계값보다 크면 value, 작으면 0 할당.
  - cv2.THRESH_BINARY_INV : 픽셀 값이 경게값보다 크면 0, 작으면 value 할당.
  - cv2.THRESH_TRUNC : 픽셀 값이 경계값보다 크면 threshold_value, 작으면 픽셀 값 할당.
  - cv2.THRESH_TOZERO : 픽셀 값이 경계값보다 크면 픽셀 값, 작으면 0을 할당
  - cv2.THRESH_TOZERO_INV : 픽셀 값이 경계값보다 크면 0, 작으면 픽셀 값을 할당.
  - cv2.THRESH_OTSU : 이에 더해서 OTSU 이진화가 있으며 cv2.theshold() 함수의 flag값에 더해주면 적용한 결과를 리턴합니다. - 간단히, OTSU Binarization은 두 봉우리 사의 값을 선택하여 더 개선되게 이진화를 해줍니다.(즉, 분산을 최소로 한다.) OTSU 이진화에 대한 설명은 아래 링크의 설명을 보면 이해하기 쉽습니다.

ret : 사용된 임계값

thresh : 출력된 이미지

더불어, cv2.threshold의 경우 임계값을 이미지 전체에 적용하여 처리학 떄문에 하나의 이미지에 음영이 다르면 일부 영역이 모두 흰색 또는 검정색으로 나와버린다는 문제가 있는데 이에 대하여 `cv2.adaptiveThreshold()`가 나왔습니다. 

또한, OTSU 이진화에 대한 설명도 읽어보시는 것을 추천드립니다.

이에 대한 내용은 [이미지 임계처리](https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html#cv2.adaptiveThreshold)에 잘 정리되어있습니다.


```python
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# 이미지만 출력하기 위해아 인덱스로 1을 입력하였습니다.
cv2_imshow(thresh)
```


![3번](https://user-images.githubusercontent.com/53552847/120942690-90ce5e00-c765-11eb-96f3-12e2e6edd1c6.png)


## 4. cv2.Canny

Canny는 가장자리를 검출하는 알고리즘 중에 하나입니다.

가장자리란 가장 바깥 부분의 둘레의 의미이며 객체의 테두리로 볼 수도 있다. 이미지 상에서는 전경과 배경이 구분되는 지점이며, 전경과 배경 사이에서 **밝기가 큰 폭으로 변하는 지점**이 가장자리가 됩니다.

가장자리를 찾기 위해서는 미분과 기울기 연산을 수행하고, 이미지 상에서 픽셀의 밝기 변화율이 높은 경계선을 찾습니다.

가장 자리를 검출하는 알고리즘은 `Sobel`, `Laplacian`, `Canny`이 있으며 본 장은 Canny에 대해서 설명하며 다른 알고리즘에 대한 설명은 아래 링크를 참조하면 이해하기 쉽습니다.

[Python OpenCV 강좌 : 제 14강 - 가장자리 검출](https://076923.github.io/posts/Python-opencv-14/)  
[Canny Edge Detection](https://m.blog.naver.com/samsjang/220507996391)  


Canny edge는 Laplace 필터 방식을 개선한 방식으로 x, y에 대한 1차 미분을 계산한 다음, 네 방향으로 미분을 한다.

네 방향으로 미분한 결과로 극댓값을 갖는 지점들이 가장자리를 가지게 된다.

`Sobel`, `Laplacian`보다 성능이 좋고 노이즈에 민감하지 않습니다.



```
canny = cv2.Canny(img, threshold_1, threshold_2)
```

img : 입력 이미지 ( numpy array )

threshold_1 : 하위 임계값. 픽셀이 하위 임계값보다 낮은 경우 가장자리로 고려하지 않는다.

threshold_2 : 상위 임계값. 픽셀 상위 임계값보다 높은 경우 가장자리로 간주한다.


```python
canny = cv2.Canny(img, 50, 50)
cv2_imshow(canny)
```


![4번](https://user-images.githubusercontent.com/53552847/120942692-90ce5e00-c765-11eb-8746-b4dcfd0b8a9a.png)


## 5. 모폴로지 연산 (cv2.getStructuringElement, cv2.dilate, cv2.erode, cv2.mophologyEx)

모폴로지 변환은 영상이나 이미지를 형태학적 관점에서 접근하는 기법이다.

모폴로지 변환은 주로 영상 내 픽셀값 대체에 사용된다. 이를 응용하여 노이즈 제거, 요소 결합 및 분리, 강도 피크 검출 등에 이용할 수 있다.

기본적인 모폴로지 변환으로 사용하는 것이 `dilate`, `erode`이며 이는 이미지와 커널의 convolution연산이고 이 두 연산을 기반으로 복잡하고 다양한 모폴로지 연산을 구현할 수 있다.

`cv2.mophologyEx`는 grayscale이나 다중 채널 이미지를 사용할 경우 더 복자한 연산을 필요로 하는데 이 때 사용하면 더 우수한 결과를 얻을 수 있다.


## 6. cv2.getStructuringElement - 모폴로지 연산




```
kernel = cv2.getStructuringElement(shape, ksize, anchor)
```

shape : 구조화 요소 커널의 모양
- cv2.MORPH_CROSS : 십자가형
- cv2.MORPH_ELLIPSE : 타원형
- cv2.MORPH_RECT : 직사각형

kisze : 구조화 요소 커널의 크기. 이 떄, 커널의 크기가 너무 작다면 커널의 형태는 영향을 받지 않는다.

anchor : 커널의 중심 위치를 나타낸다. 필수 매개변수가 아니며 설정하지 않을 경우 사용되는 함수에서 값이 결정된다.


```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
```

## 7. cv2.dilate - 모폴로지 연산


Dilation은 커널 영역 안에 존재하는 모든 픽셀 값을 커널 내부의 극댓값으로 대체한다. 즉, 구조 요소를 활용해 이웃한 픽셀들을 최대 픽셀 값으로 대체한다.

팽창 연산을 적용하면 **어두운 영역이 줄어들고 밝은 영역이 늘어난다.**

커널의 크기 혹은 반복 횟수에 따라서 밝은 영역이 늘어나 speckle이 커지며 객체 내부의 holes이 사라진다.

dilate 연산의 경우 **노이즈 제거 후 줄어든 크기를 복구하조가 할 때 주로 사용**한다.

```
dilate = cv2.dilate(img, kernel, anchor, iterations)
```

img : 입력 이미지. 채널 수는 상관 없으나 다음과 같은 이미지 데이터만 사용할 수 있다. (CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)

kernel : 팽창에 사용할 구조화 요소 커널. `5. cv2.getStructuringElement()`를 사용하여 생성할 수 있다.

anchor : 구조화 요소 안에서 사용할 기준점. default 값으로 (-1, -1)이 지정되어 있어 구조화 요소 중심을 기준으로 잡는다.

iterations : dilate를 반복할 횟수를 지정한다.


```python
dilate = cv2.dilate(thresh, kernel, anchor = (-1, -1), iterations = 3)
cv2_imshow(dilate)
```


![5번](https://user-images.githubusercontent.com/53552847/120942693-9166f480-c765-11eb-86de-5900e65e71a2.png)

## 8. cv2.erode - 모폴로지 연산

erosion은 커널 영역 안에 존재하는 모든 픽셀의 값을 커널 내부의 극솟값으로 대채한다. 즉, 구조 요소를 활용하여 이웃한 픽셀을 최소 픽셀값으로 대체한다.

침식 연산을 적용하면 **밝은 영역이 줄어들고 어두운 영역이 늘어난다.**

커널의 크기나 반복 횟수에 따라 어두운 영역이 늘어나 Speckle이 사라지며, 객체 내부의 holes이 커집니다.

erode 연산은 주로 **노이즈 제거에 사용**합니다.

```
erode = cv2.erode(dilate, kernel, anchor, iterations)
```

dilate : 입력이미지 이며 흔히 dilate를 한 후에 dilate된 이미지를 입력값으로 사용한다. 채널 수는 상관 없으나 다음과 같은 이미지 데이터만 사용할 수 있다. (CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)

kernel : 팽창에 사용할 구조화 요소 커널. `5. cv2.getStructuringElement()`를 사용하여 생성할 수 있다.

anchor : 구조화 요소 안에서 사용할 기준점. default 값으로 (-1, -1)이 지정되어 있어 구조화 요소 중심을 기준으로 잡는다.

iterations : erode를 반복할 횟수를 지정한다.


```python
erode = cv2.erode(dilate, kernel, anchor = (-1, -1), iterations = 3)
cv2_imshow(erode)
```


![6번](https://user-images.githubusercontent.com/53552847/120942694-9166f480-c765-11eb-82be-e5f133ff4255.png)

## 9. cv2.morphologyEx - 모폴로지 연산

cv2.morphologyEx() 함수는 침식과 팽창 뿐만 아니라 다양한 연산을 지원한다.



```
detect = cv2.morphologyEx(img, flag, kernel, iterations)
```
img : 입력 이미지

flag : 모폴로지 연산 플래그
  - cv2.MORPH_DILATE   : 팽창 연산
  - cv2.MORPH_ERODE    : 침식 연산
  - cv2.MORPH_OPEN     : 열림 연산
  - cv2.MORPH_CLOSE    : 닫힘 연산
  - cv2.MORPH_GRADIENT : 그래디언트 연산
  - cv2.MORPH_TOPHAT   : 탑햇 연산
  - cv2.MORPH_BLACKHAT : 블랙햇 연산
  - cv2.MORPH_HITMISS  : 힛미스 연산

kernel : 모폴로지 연산에 사용할 구조화 요소 커널. `5. cv2.getStructuringElement()`를 사용하여 생성할 수 있다.

iterations : 모폴로지 연산을 반복할 횟수를 지정한다.

모폴로지 연산 플래그에 대한 자세한 설명은 [Python OpenCV 강좌 : 제 27강 - 모폴로지연산](https://076923.github.io/posts/Python-opencv-27/)을 보면 식들을 확인할 수 있습니다.


```python
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
vertical_detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                          vertical_kernel, iterations = 2)
cv2_imshow(vertical_detect)
```


![7번](https://user-images.githubusercontent.com/53552847/120942695-91ff8b00-c765-11eb-9882-fce3819b6fa5.png)

## 10. cv2.findContours

cv2.findContours()은 영상이나 이미지의 윤곽선을 검출하기 위해 사용한다.

cv.findContours()를 이용하여 **이진화 이미지**에서 윤곽선을 검색한다.

cv2.findContours()의 경우 Suzuki85라는 알고리즘을 활용하여 이미지에서 Contour를 찾는 함수이며 이는 **원본 이미지를 변경시키기 때문에 향후 원본 이미지를 활용하기 위해서는 원본 이미지의 복사본을 가지고 진행**하도록 한다.

더불어, OpenCV에서 Contour 찾기는 검정색 배경에서 흰색 물체를 찾는 것과 비슷하므로 Contour를 찾고자 할 때는 대상은 흰색, 배경은 검정색으로 변경해야 함을 꼭 기억하도록 하자.

```
cnts = cv2.findContours(img, mode, method)
```
image : 8-bit single-channel image or binary image

mode : contours를 찾는 방법
  - cv2.RETR_EXTERNAL : contours line 중 가장 바깥쪽 line만 찾는다.
  - cv2.RETR_LIST : 모든 contours line을 찾지만, hierachy 관계를 구성하지 않는다.
  - cv2.RETR_CCOMP : 모든 contours line을 찾고, hierachy 관계는 2-level로 구성한다.
  - cv2.RETR_TRESS : 모든 contours line을 찾고 모든 hierachy 관계를 구성한다.

method : contours를 찾을 때 사용하는 근사치 방법.
  - cv2.CHAIN_APPROX_NONE : 모든 contours point를 저장
  - cv2.CHAIN_APPROX_SIMPLE : contours line을 그릴 수 있는 point만 저장.
  - cv2.CHAIN_APPROX_TC89_L1 : 프리먼 체인 코드에서의 윤곽선으로 적용한다.
  - cv2.CHAIN_APPROX_TC89_KCOS : 프리먼 체인 코드에서의 윤곽선으로 적용한다.

return : contour, hierarchy

contour : Numpy 구조의 배열로 검출된 윤곽선의 지점.

hierarchy : 윤곽선의 계층 구조. 각 윤곽선에 해당하는 속성 정보들이 담겨있다.


```python
cnts = cv2.findContours(vertical_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts
```




    ([array([[[ 832, 1938]],
      
             [[ 832, 1966]],
      
             [[ 833, 1966]],
      
             [[ 833, 1938]]], dtype=int32), array([[[ 678, 1938]],
      
             [[ 678, 1966]]], dtype=int32), array([[[ 527, 1938]],
      
             [[ 527, 1966]]], dtype=int32), array([[[1110, 1480]],
      
             [[1110, 1576]],
      
             [[1111, 1575]],
      
             [[1111, 1481]]], dtype=int32), array([[[984, 522]],
      
             [[984, 550]],
      
             [[985, 550]],
      
             [[985, 522]]], dtype=int32), array([[[ 929,  504]],
      
             [[ 929, 2167]],
      
             [[ 930, 2167]],
      
             [[ 930,  504]]], dtype=int32), array([[[1439,  495]],
      
             [[1439,  523]]], dtype=int32), array([[[1265,  495]],
      
             [[1265,  523]]], dtype=int32), array([[[335, 495]],
      
             [[335, 523]]], dtype=int32), array([[[239, 495]],
      
             [[239, 523]]], dtype=int32), array([[[210, 495]],
      
             [[210, 523]]], dtype=int32), array([[[685, 460]],
      
             [[685, 488]]], dtype=int32), array([[[462, 460]],
      
             [[462, 488]]], dtype=int32), array([[[1598,  449]],
      
             [[1598, 2167]],
      
             [[1599, 2166]],
      
             [[1599,  449]]], dtype=int32), array([[[1354,  449]],
      
             [[1354, 1891]],
      
             [[1355, 1892]],
      
             [[1355, 1921]],
      
             [[1354, 1922]],
      
             [[1354, 2167]],
      
             [[1355, 2167]],
      
             [[1355,  449]]], dtype=int32), array([[[1102,  449]],
      
             [[1102, 2167]],
      
             [[1103, 2167]],
      
             [[1103, 1922]],
      
             [[1102, 1921]],
      
             [[1102, 1892]],
      
             [[1103, 1891]],
      
             [[1103,  449]]], dtype=int32), array([[[ 771,  449]],
      
             [[ 771, 2167]],
      
             [[ 773, 2167]],
      
             [[ 773,  559]],
      
             [[ 772,  558]],
      
             [[ 772,  449]]], dtype=int32), array([[[ 551,  449]],
      
             [[ 551, 2167]],
      
             [[ 552, 2167]],
      
             [[ 552,  449]]], dtype=int32), array([[[ 417,  449]],
      
             [[ 417, 2167]],
      
             [[ 418, 2167]],
      
             [[ 418,  449]]], dtype=int32), array([[[ 283,  449]],
      
             [[ 283, 2167]],
      
             [[ 284, 2167]],
      
             [[ 284,  449]]], dtype=int32), array([[[ 117,  449]],
      
             [[ 117, 2167]],
      
             [[ 119, 2167]],
      
             [[ 119,  449]]], dtype=int32), array([[[  47,  449]],
      
             [[  47, 2166]],
      
             [[  48, 2167]],
      
             [[  48,  449]]], dtype=int32), array([[[626, 272]],
      
             [[626, 300]],
      
             [[627, 300]],
      
             [[627, 272]]], dtype=int32), array([[[1082,  268]],
      
             [[1082,  321]],
      
             [[1083,  322]],
      
             [[1084,  322]],
      
             [[1084,  320]],
      
             [[1085,  319]],
      
             [[1085,  269]],
      
             [[1083,  269]]], dtype=int32), array([[[1094,  264]],
      
             [[1094,  330]],
      
             [[1097,  330]],
      
             [[1097,  317]],
      
             [[1098,  316]],
      
             [[1098,  265]],
      
             [[1097,  264]]], dtype=int32), array([[[1026,  264]],
      
             [[1026,  303]],
      
             [[1028,  303]],
      
             [[1028,  301]],
      
             [[1029,  300]],
      
             [[1029,  265]],
      
             [[1028,  264]]], dtype=int32), array([[[955, 264]],
      
             [[955, 330]],
      
             [[957, 330]],
      
             [[957, 323]],
      
             [[958, 322]],
      
             [[958, 265]],
      
             [[957, 264]]], dtype=int32), array([[[892, 264]],
      
             [[892, 303]],
      
             [[894, 303]],
      
             [[894, 301]],
      
             [[895, 300]],
      
             [[895, 265]],
      
             [[894, 264]]], dtype=int32), array([[[826, 264]],
      
             [[826, 303]],
      
             [[828, 303]],
      
             [[828, 301]],
      
             [[829, 300]],
      
             [[829, 265]],
      
             [[828, 264]]], dtype=int32), array([[[755, 264]],
      
             [[755, 300]],
      
             [[757, 300]],
      
             [[757, 299]],
      
             [[758, 298]],
      
             [[758, 265]],
      
             [[757, 264]]], dtype=int32), array([[[588, 264]],
      
             [[588, 330]],
      
             [[590, 330]],
      
             [[590, 323]],
      
             [[591, 322]],
      
             [[591, 265]],
      
             [[590, 264]]], dtype=int32), array([[[522, 264]],
      
             [[522, 330]],
      
             [[524, 330]],
      
             [[524, 323]],
      
             [[525, 322]],
      
             [[525, 265]],
      
             [[524, 264]]], dtype=int32)], array([[[ 1, -1, -1, -1],
             [ 2,  0, -1, -1],
             [ 3,  1, -1, -1],
             [ 4,  2, -1, -1],
             [ 5,  3, -1, -1],
             [ 6,  4, -1, -1],
             [ 7,  5, -1, -1],
             [ 8,  6, -1, -1],
             [ 9,  7, -1, -1],
             [10,  8, -1, -1],
             [11,  9, -1, -1],
             [12, 10, -1, -1],
             [13, 11, -1, -1],
             [14, 12, -1, -1],
             [15, 13, -1, -1],
             [16, 14, -1, -1],
             [17, 15, -1, -1],
             [18, 16, -1, -1],
             [19, 17, -1, -1],
             [20, 18, -1, -1],
             [21, 19, -1, -1],
             [22, 20, -1, -1],
             [23, 21, -1, -1],
             [24, 22, -1, -1],
             [25, 23, -1, -1],
             [26, 24, -1, -1],
             [27, 25, -1, -1],
             [28, 26, -1, -1],
             [29, 27, -1, -1],
             [30, 28, -1, -1],
             [31, 29, -1, -1],
             [-1, 30, -1, -1]]], dtype=int32))



cv2.findContours에 대한 자세한 내용은 [Image Contours](https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html)을 참고하면 보다 쉽게 이해할 수 있다.

더불어, 계층구조에 대한 설명이 [Python OpenCV 강좌 : 제 21강 - 윤곽선 검출](https://076923.github.io/posts/Python-opencv-21/)에 있으므로 참고하면 좋습니다.

## 11. cv2.drawContours

cv2.drawContours()을 이용하여 검출된 윤곽선을 그린다.

drawContours 역시 findContours와 마찬가지로 기존의 이미지에 draw하기 때문에 복사하여 사용하는 것이 좋다.

```
cv2.drawContours(img, contours, contoursIdx, color, thickness)
```
img : 원본 이미지

contours : 윤곽선 정보

contourIdx : contours list type에서 몇 번째 contours line을 그릴 것인지. -1이면 전체를 그린다.

color : contours line color

thickness : contours line의 두꼐. 음수를 넣었을 경우 contours line의 내부를 채운다.

return : 윤곽선을 그린 image


```python
clean = img.copy()
cnts = cv2.findContours(vertical_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1] # 윤곽선의 값만 가져오기 위해 실행한다.

for c in cnts :
  cv2.drawContours(clean, [c], -1, 0, 3)

cv2_imshow(clean)
```


![8번](https://user-images.githubusercontent.com/53552847/120942696-91ff8b00-c765-11eb-9b6b-d1055dfcaa78.png)

## 12. cv2.boundingRect

윤곽선을 둘러싸는 박스를 그려준다.

흔히 cv2.boundingRect() 함수는 그림을 그릴 때 그 그림이 들어갈 박스값의 좌표를 잡아준다. 하지만, 본 코드의 이미지에서는 cell을 object로 인식하여 진행하는 과정이므로 boundingRect와 contour의 값이 거의 유사함을 알 수 있다.

```
x, y, w, h = cv2.boudndingRect(cnt)
```
cnt : 윤곽선정보

x, y, w, h : 각각 왼쪽위의 x, y좌표, 가로선의 길이 w, 세로선의 길이 h를 나타낸다.


```python
boundingbox = cv2.boundingRect(cnts[0])
print(boundingbox[1]) # y값의 좌표만 출력해보도록 한다.
```

    1938


이미지 Contour에 대한 응용으로 cv2.boundingRect를 활용하며 cv2.boundingRect 외에 여러가지 응용함수가 있으며 이는 [이미지 Contour 응용2](https://m.blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=samsjang&logNo=220517391218)를 참고하면 좋다.

## 13. cv2.contourArea

외각선이 감싸는 영역의 면적을 반환한다.



```
area = cv2.contourArea(c)
```
c : 윤곽선 하나의 정보(contour에서의 하나의 값)

area : 면적값


```python
area = cv2.contourArea(cnts[0])
print(area)
```

    28.0


## 14. cv2.line

cv2.line() 함수를 활용하여 두 좌표를 있는 선을 그을 수 있다.

cv2.line() 함수는 기존의 이미지에 바로 그림을 그리므로 기존의 이미지를 보존하고 싶을 경우 copy해서 사용하는 것이 좋다.

```
cv2.line(img, pt1, pt2, color, thickness)
```
img : 이미지 파일

pt1 : 시작점 좌표 (x, y)

pt2 : 종료점의 좌표 (x, y)

color : 색상 (r, g, b). 0 ~ 255

thickness : 선의 두께 (default : 1)


```python
clean = img.copy()
cv2.line(clean, (-50000, 1000), (50000, 1000), (0,0,0), 1)
cv2.line(clean, (-50000, 500), (50000, 500), (0,0,0), 1)
cv2.line(clean, (-50000, 1500), (50000, 1500), (0,0,0), 1)
cv2_imshow(clean)
```


![9번](https://user-images.githubusercontent.com/53552847/120942697-92982180-c765-11eb-93e1-65524174a416.png)

Line 뿐만 아니라 다양한 도형을 그리기 위한 정보는 [OpenCV 1. 도형 그리기 점 선 (Python)](https://copycoding.tistory.com/145) 을 참고하면 보다 많은 정보를 얻을 수 있다.

## 15. cv2.GaussianBlur



cv2.GaussianBlur는 기존의 cv2.blur에서 사용했던 평균값 필터 블러링의 단점을 보완하기 위해 사용된다.

평균값 필터란 영상의 특정 좌표 값을 주변 픽셀 값들의 산술 평균으로 설정하는 방법이다. 이를 사용하는 이유는 픽셀들 간의 grayscale 값의 변화가 줄어들어 날카로운 edge들이 무뎌지고, 영상에 있는 noise의 영향이 사라지는 효과가 있어 자주 사용됩니다.

GaussianBlur의 경우 기존의 평균값 필터에 가우시안 함수를 사용하여 거리에 따른 가중치를 함꼐 사용합니다.

보다 자세한 설명은 [영상에 블러링(가우시안 필터) 적용하기 - cv2.GaussianBlur](https://deep-learning-study.tistory.com/144)를 참고하면 이해가 쉽습니다.

```
dst = cv2.GaussianBlur(src, ksize, sigmaX, sigmaY = None)
```
src : 입력 영상 혹은 이미지. 각 채널 별로 처리된다.

ksize : 가우시안 커널 크기. (0,0)을 지정하면 sigma 값에 의해 자동으로 결정된다.

sigmaX : x방향 sigma

sigmaY : y방향 sigma, 0일 경우 sigmaX와 같게 설정된다.


```python
blur_img = cv2.GaussianBlur(img, (7,7), 0)
cv2_imshow(blur_img)
```


![10번](https://user-images.githubusercontent.com/53552847/120942699-92982180-c765-11eb-9dff-d9c493481c8d.png)


## 참고 URL


https://076923.github.io/posts/Python-opencv-10/  
https://zzsza.github.io/data/2018/01/23/opencv-1/  
https://m.blog.naver.com/samsjang/220504782549  
https://bkshin.tistory.com/entry/OpenCV-8-%EC%8A%A4%EB%A0%88%EC%8B%9C%ED%99%80%EB%94%A9Thresholding  
https://076923.github.io/posts/Python-opencv-14/  
https://m.blog.naver.com/samsjang/220507996391  
https://dsbook.tistory.com/203  
https://076923.github.io/posts/Python-opencv-26/  
https://076923.github.io/posts/Python-opencv-27/  
https://m.blog.naver.com/samsjang/220516697251  
https://076923.github.io/posts/Python-opencv-21/  
https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html  
https://m.blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=samsjang&logNo=220517391218  
https://copycoding.tistory.com/145  
https://deep-learning-study.tistory.com/144


```python

```
