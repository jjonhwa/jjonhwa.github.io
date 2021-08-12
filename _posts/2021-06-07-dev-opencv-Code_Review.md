---
layout: post
title: "Code Review(Drawing_Using_OpenCV)"
categories: dev
tags: opencv
comments: true
---

본 장은 [Drawing_Using_OpenCV](https://github.com/jjonhwa/Drawing_Using_OpenCV)에 대한 **코드 리뷰 페이지**이며 궁극적으로는 Drawing_Using_OpenCV를 활용하여 Table Extraction을 진행하기위한 전처리 과정입니다. **Table Extraction**에 대한 내용도 관심이 있다면 [Table_Extraction_Kor-benchmark](https://github.com/jjonhwa/Table_Extraction_Kor-benchmark)에 한 번 들러주세요.

더불어, 본 장에서 활용한 OpenCV 코드 설명을 보다 상세히 [Code Explanation(Drawing_Using_OpenCV)](https://jjonhwa.github.io/2021-06-06-Code_Explanation/)에 업로드해두었습니다. 참고하며 이해하시기 바랍니다.

## 목차
[1. image_scale 함수](#1-image_scale-함수)  
[2. cut_image 함수](#2-cut_image-함수)  
[3. search_x 함수](#3-search_x-함수)  
[4. remove_horizontal & remove_vertical 함수](#4-remove_horizontal--remove_vertical-함수)  
[5. dilate_and_erode 함수](#5-dilate_and_erode-함수)  
[6. preprocess_image 함수](#6-preprocess_image-함수)  
[7. draw_line 함수](#7-draw_line-함수)    
[8. Line Drawing](#8-line-drawing)


```python
img = cv2.imread(image, cv2.IMREAD_COLOR)
cv2_imshow(img)
```


![1번](https://user-images.githubusercontent.com/53552847/120977556-69e84a00-c7ae-11eb-9428-e32a4970b158.png)


**Note :**  본 장에서 사용한 OpenCV 코드에 보다 대한 자세한 설명은 https://jjonhwa.github.io/2021-06-06-Code_Explanation/에서 확인할 수 있다.

## 1. image_scale 함수

이미지를 가공하기 위한 가장 기본적인 전처리 단계.


``` 
def image_scale(img) :
  '''
  img : numpy array형태의 image
  return : scaled image
  '''
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  return thresh

def image_scale_sub(img) :
  '''
  img : numpy array형태의 image
  return : scaled image
  '''
  canny = cv2.Canny(img, 50, 50)
  thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU)[1]
  return thresh
```

- **grayscale** : gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
- **이미지 임계처리** : thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

grayscale - threshold 처리가 가장 기본적이며 본 장의 경우 데이터프레임기반 이미지를 가공하기 위한 단계로서 Table에서 선이 옅은 경우 다음의 edge알고리즘을 사용하여 전처리를 진행한다.

- **Edge 알고리즘을 활용한 line detect** : canny = cv2.Canny(img, 50, 50)
- **이미지 임계처리** : cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU)[1]



```python
scale_image = md.image_scale(img)
cv2_imshow(scale_image)
```


![2번](https://user-images.githubusercontent.com/53552847/120977561-6b197700-c7ae-11eb-9cbd-7df878224845.png)


## 2. cut_image 함수

이미지에서 원하는 길이만 자르기 위해 만든 함수.

본 장에서 활용하는 예시 데이터의 경우 Table이미지가 Table만 있는 것이 아니라 다른 Text들이 섞여 있기 때문에 Table만 가져와 Drawing하기 위하여 본 코드를 사용하였습니다.

만약, Table이미지에서 필요한 부분을 자르거나 발췌할 때 본 코드를 응용하여 사용할 수 있습니다.


```
def cut_image(scale_img, threshold = 800) :
  '''
  scale_img : 임계처리된 이미지(thresh)
  threshold : 이미지를 자르기 위한 선들 사이의 간격
  return : 원하는 길이만큼 잘려진 이미지
  '''
  horizontal_kernel = cv2.getSTructuringElement(cv2.MORPH_RECT, (81,1))
  detect_horizontal = cv2.morphologyEx(scale_img, cv2.MORPH_OPEN, horizontal_kernel, iterations = 3)
  cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  for i in range(len(cnts)) :
    if i == 0 :
      continue
    first_line = cv2.boundingRect(cnts[i-1])[1]
    second_line = cv2.boundingRect(cnts[i])[1]
    
    if abs(first_line - second_line) >= threshold : 
      start_line = second_line-5
      break
  clean = scale_img[start_line:, :]
  return start_line, clean
```

- 구조화 커널의 생성(수평선) : horizontal_kernel = cv2.getSTructuringElement(cv2.MORPH_RECT, (81,1))
- 열림연산을 활용한 모폴로지 변환 : detect_horizontal = cv2.morphologyEx(scale_img, cv2.MORPH_OPEN, horizontal_kernel, iterations = 3)
- 이미지 윤곽선 검출 : cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
- 계층구조를 제외한 윤곽선만 입력 : cnts = cnts[0] if len(cnts) == 2 else cnts[1]
- 시작점 도출 : for문
- 인덱싱을 활용하여 이미지 자르기 : clean = scale_img[start_line:, :]



```python
start_line, scale_cut_image = md.cut_image(scale_image)
print(start_line)
cv2_imshow(scale_cut_image)
```

    554



![3번](https://user-images.githubusercontent.com/53552847/120977545-6785f000-c7ae-11eb-883a-3aaaf8bdca4f.png)


## 3. search_x 함수

Line Drawing을 깔끔하게 하기 위한 x좌표의 최대 최소를 구하는 함수이다.

이를 응용하여 y좌표의 최대 최소 역시 구할 수 있다.

```
def search_x(scale_image) :
  '''
  scale_image : 임계처리된 이미지(thresh)
  return : 최대 x, 최소 x 좌표값
  '''

  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
  detect_vertical = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                    vertical_kernel, iterations = 3)
  cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  x_list = []
  for i in range(len(cnts)) :
    x_list.append(list(cv2.boundingRect(cnts[i][0])))
  
  tmp = pd.DataFrame(x_list)
  max_x = np.max(tmp[0])
  min_x = np.min(tmp[0])
  return min_x, max_x
```

- 구조화 커널의 생성(수직선) : vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
- 열림연산을 활용한 모폴로지 변환 : detect_vertical = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN, vertical_kernel, iterations = 3)
- 이미지 윤곽선 검출 : cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
- 계층구조를 제외한 윤곽선만 입력 : cnts = cnts[0] if len(cnts) == 2 else cnts[1]
- 수직선들의 x좌표 값 입력 : x_list
- x좌표의 최대 최소 검출 : max_x, min_x


```python
min_x, max_x = md.search_x(scale_image)
print(min_x, max_x)
```

    47 1598


## 4. remove_horizontal & remove_vertical 함수

본 장에서는 Text기반 수평선 Line Drawing을 활용하여 Table without cell에서 cell을 만들어준다.

여기에서 Text만을 기준으로 Line Drawing을 하기 위하여 수직, 수평선을 삭제하는 작업을 진행한다.

```
def remove_horizontal(scale_image) :
  '''
  scale_image : 임계처리된 이미지(tresh)
  return : 임계처리된 이미지에서 수직선이 삭제된 이미지
  '''
  clean = scale_image.copy()
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
  detect_horizontal = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                        horizontal_kernel, iterations  = 2)
  cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  for c in cnts :
    cv2.drawContours(clean, [c], -1, 0, 3)
      
  return clean

def remove_vertical(scale_image) :
  '''
  scale_image : 임계처리된 이미지(tresh)
  return : 임계처리된 이미지에서 수평선이 삭제된 이미지
  '''
  clean = scale_image.copy()
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
  detect_vertical = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                      vertical_kernel, iterations = 3)
  cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  for c in cnts :
      cv2.drawContours(clean, [c], -1,  0, 3)
      
  return clean
```

- 수직, 수평선의 구조화 커널 생성 : cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) - (수직선의 경우 위 코드와 같이 ksize를 (1, 15)로 변경)
- 열림연산을 활용한 모폴로지 변환 : detect_horizontal = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN, horizontal_kernel, iterations  = 2)
- 이미지 윤곽선 검출 : cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
- 계층구조를 제외한 윤곽선만 입력 : cnts = cnts[0] if len(cnts) == 2 else cnts[1]
- 윤곽선을 지워준다 : for문(cv2.drawContours) - scaleimage에 vertical line을 그리는 형식으로 하여 실제로는 윤곽선을 지우는 효과를 나타낸다.(scale되어 있지 않을 경우 윤곽선을 그린다.)


```python
scale_cut_image = md.remove_horizontal(scale_cut_image)
scale_cut_image = md.remove_vertical(scale_cut_image)
cv2_imshow(scale_cut_image)
```


![4번](https://user-images.githubusercontent.com/53552847/120977547-68b71d00-c7ae-11eb-94c6-a436fe4a10f0.png)


## 5. dilate_and_erode 함수

Text를 기준으로 Line Drawing을 진행하기 위하여 수직, 수평선이 지워지고 Text만 남아있는 image에서 전처리 과정(dilate, erode)를 수행하고 윤곽값을 찾아주는 함수이다.

```
def dilate_and_erode(scale_image, dil_iterations = 5, erode_iterations = 5) :
  '''
  scale_image : 임계처리된 이미지(thresh)
  dil_iterations : dilate의 반복횟수
  erode_iterations : erode의 반복횟수
  return : Text의 윤곽Box 값
  '''
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
  dilate = cv2.dilate(scale_image, kernel, anchor = (-1, -1), iterations = dil_iterations)
  erode = cv2.erode(dilate, kernel, anchor = (-1, -1), iterations = erode_iterations)
  
  cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  return cnts
```
- 구조화 커널 생성(2,2 Box) : kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
- 팽창 연산을 활용한 모폴로지 변환 : dilate = cv2.dilate(scale_image, kernel, anchor = (-1, -1), iterations = dil_iterations)
- 침식 연산을 활용한 모폴로지 변환 : erode = cv2.erode(dilate, kernel, anchor = (-1, -1), iterations = erode_iterations)
- 이미지 윤곽 Box값 검출 : cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
- 계층구조를 제외한 윤곽선만 입력 : cnts = cnts[0] if len(cnts) == 2 else cnts[1]


```python
contour = md.dilate_and_erode(scale_cut_image, 5, 2)
contour # contour에 대한 출력은 생략하도록 한다.
```

## 6. preprocess_image 함수

최종으로 dilate_and_erode에서 검출된 contour를 바탕으로 우리가 정말로 그려줄 Line에 대한 좌표값을 DataFrame형식으로 출력하는 함수

```
def preprocess_image(contour) :
  '''
  contour : Text의 박스 윤곽값
  return : 실제로 Line Drawing을 할 y값을 포함한 좌표 테이블
  '''
  final_list = []
  for c in contour : 
    final_list.append(list(cv2.boundingRect(c)))
      
  final_data = pd.DataFrame()
  for i in range(len(final_list)) :
    new_row = final_list[i]
    new_row = pd.DataFrame(new_row).T
    
    final_data = pd.concat([final_data, new_row])
    
  final_data.reset_index(drop = True, inplace = True)
  final_data.columns = ['x', 'y', 'w', 'h']
  
  tmp = final_data.groupby('y').agg({'h' : 'max'})
  temp = tmp.reset_index()
  
  
  drop_list = []
  for i in range(len(temp)) :
    if i == 0 :
      continue
    if abs(temp['y'][i-1] - temp['y'][i]) <= 10 and \
      abs(temp['h'][i-1] - temp['h'][i]) <= 25:
      if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i]:
        drop_list.append(i)
      else :
        drop_list.append(i-1)
              
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)
  
  drop_list = []
  for i in range(len(temp)) :
    if i == 0 :
      continue
    if abs(temp['y'][i-1] - temp['y'][i]) <= 15 and \
      abs(temp['h'][i-1] - temp['h'][i]) <= 25:
      if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i] :
        drop_list.append(i)
      else :
        drop_list.append(i-1)
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)

  drop_list = []
  for i in range(len(temp)) :
    if i == 0 :
      continue
    if abs(temp['y'][i-1] - temp['y'][i]) <= 25 and \
      abs(temp['h'][i-1] - temp['h'][i]) <= 25:
      if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i] :
        drop_list.append(i)
      else :
        drop_list.append(i-1)
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)
  
  temp['yh'] = temp['y'] + temp['h']
  temp = temp.sort_values('yh')

  drop_list = []
  for i in range(len(temp['yh'])) :
    if i == 0 :
      continue
    if abs(temp['yh'][i-1] - temp['yh'][i]) <= 25 :
      drop_list.append(i-1)
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)

  temp = temp.drop(['yh'], axis = 1)
  final = pd.merge(temp, final_data)
  return final
```

- 수평선을 그리기 위한 Text들의 박스값의 아래 y값 : 맨 처음 temp를 구하는 과정
- 겹치거나 간격이 좁을 경우 맨 아래 y값만을 출력 : drop_list를 활용한 box값 drop - for문의 반복


```python
final = md.preprocess_image(contour)
final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>h</th>
      <th>x</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>28</td>
      <td>437</td>
      <td>99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23</td>
      <td>28</td>
      <td>166</td>
      <td>74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>85</td>
      <td>24</td>
      <td>652</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>118</td>
      <td>27</td>
      <td>646</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>118</td>
      <td>27</td>
      <td>304</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>118</td>
      <td>27</td>
      <td>192</td>
      <td>50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>118</td>
      <td>27</td>
      <td>168</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>166</td>
      <td>29</td>
      <td>665</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>260</td>
      <td>27</td>
      <td>672</td>
      <td>37</td>
    </tr>
    <tr>
      <th>9</th>
      <td>260</td>
      <td>27</td>
      <td>622</td>
      <td>21</td>
    </tr>
    <tr>
      <th>10</th>
      <td>307</td>
      <td>27</td>
      <td>647</td>
      <td>37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>354</td>
      <td>27</td>
      <td>702</td>
      <td>26</td>
    </tr>
    <tr>
      <th>12</th>
      <td>402</td>
      <td>26</td>
      <td>646</td>
      <td>38</td>
    </tr>
    <tr>
      <th>13</th>
      <td>449</td>
      <td>26</td>
      <td>646</td>
      <td>38</td>
    </tr>
    <tr>
      <th>14</th>
      <td>496</td>
      <td>27</td>
      <td>646</td>
      <td>38</td>
    </tr>
    <tr>
      <th>15</th>
      <td>544</td>
      <td>26</td>
      <td>646</td>
      <td>38</td>
    </tr>
    <tr>
      <th>16</th>
      <td>591</td>
      <td>28</td>
      <td>1006</td>
      <td>9</td>
    </tr>
    <tr>
      <th>17</th>
      <td>685</td>
      <td>27</td>
      <td>477</td>
      <td>22</td>
    </tr>
    <tr>
      <th>18</th>
      <td>779</td>
      <td>29</td>
      <td>654</td>
      <td>32</td>
    </tr>
    <tr>
      <th>19</th>
      <td>874</td>
      <td>27</td>
      <td>74</td>
      <td>25</td>
    </tr>
    <tr>
      <th>20</th>
      <td>923</td>
      <td>28</td>
      <td>1260</td>
      <td>45</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1017</td>
      <td>27</td>
      <td>914</td>
      <td>11</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1063</td>
      <td>27</td>
      <td>666</td>
      <td>21</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1063</td>
      <td>27</td>
      <td>640</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1204</td>
      <td>28</td>
      <td>1258</td>
      <td>21</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1254</td>
      <td>28</td>
      <td>1261</td>
      <td>45</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1301</td>
      <td>28</td>
      <td>1261</td>
      <td>45</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1301</td>
      <td>28</td>
      <td>1210</td>
      <td>46</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1347</td>
      <td>29</td>
      <td>1129</td>
      <td>12</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1386</td>
      <td>32</td>
      <td>967</td>
      <td>27</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1386</td>
      <td>32</td>
      <td>814</td>
      <td>26</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1386</td>
      <td>32</td>
      <td>660</td>
      <td>31</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1386</td>
      <td>32</td>
      <td>509</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



## 7. draw_line 함수

preprocess_image로 부터 구해진 좌표값을 바탕으로 Gaussian Blur을 한 번 더 처리하여 실제로 그려줄 y값만을 출력하는 함수

```
def draw_line(image, contour, data, min_x, max_x):
  '''
  contour : 기존의 Text Box 윤곽값
  data : preprocess_image함수로 부터 생성된 좌표 테이블
  min_x : Line Drawing할 x의 최소좌표값
  max_x : Line Drawing할 x의 최대좌표값
  return : 실제로 Line Drawing할 y 좌표값
  '''
  draw_line_list = []
  for c in contour :
    for i in range(len(data)) :
      if i == len(data) - 1 :
        x = data['x'][i]
        y = data['y'][i]
        w = data['w'][i]
        h = data['h'][i]
      else :
        x_after = data['x'][i+1]
        y_after = data['y'][i+1]
        w_after = data['w'][i+1]
        h_after = data['h'][i+1]
        x_before = data['x'][i]
        y_before = data['y'][i]
        w_before = data['w'][i]
        h_before = data['h'][i]
        if abs((y_before+h_before) - (y_after + h_after)) < 25 :
          x = data['x'][i+1]
          y = data['y'][i+1]
          w = data['w'][i+1]
          h = data['h'][i+1]
        else :
          x = data['x'][i]
          y = data['y'][i]
          w = data['w'][i]
          h = data['h'][i]
      area = cv2.contourArea(c)
      if area > 40 :
        ROI = image[y:y+h, x:x+w]
        ROI = cv2.GaussianBlur(ROI, (7,7), 0)
        draw_line_list.append(y+h-2)
  return draw_line_list
```
- preprocess_image함수로부터 추출된 좌표값들을 입력 : for문 > x,y,w,h값
- 기존의 Text의 Box 윤곽값들을 바탕으로 GaussianBlur 처리를 하여 실제로 그려줄 y값만 도출 : ROI = cv2.GaussianBlur(ROI, (7,7), 0) > draw_line_list.append(y+h-2)


```python
draw_line_list = md.draw_line(img, contour, final, min_x, max_x)
draw_line_list[:10] # List 형태로 값이 많아 10개만 출력해보도록 한다.
```




    [49, 49, 107, 143, 143, 143, 143, 193, 285, 285]



## 8. Line Drawing


```python
for i in range(len(draw_line_list)) :
  y_h = draw_line_list[i]
  cv2.line(img, (min_x, y_h+start_line), (max_x, y_h+start_line), (0,0,0), 1)
```

- 실제로 기존의 image에 구해진 좌표값들을 활용해 line을 그려준다. : cv2.line(img, (min_x, y_h+start_line), (max_x, y_h+start_line), (0,0,0), 1)



```python
cv2_imshow(img)
```


![5번](https://user-images.githubusercontent.com/53552847/120977549-68b71d00-c7ae-11eb-99a0-64f52896c3e0.png)

