---
layout: post
title: "Jupyter Notebook 확장 사용"
categories: 시계열
tags: chapter1
comments: true
---
Jupyter Notebook에 대해 몰랐던 기능들에 대한 설명을 간략히 적어 놓았으며 고급 기능을 업데이트 및 설치하여 활용할 수 있도록 한다.

**NOTE :** 전체 과정에 있어서 IMAGE를 삽입하도록 한다.

## 1. Jupyter Notebook 고급세팅 : PIP & Jupyter Notebook & Jupyter Lab 얻데이트 및 확장

**1-1. Anaconda Prompt 관리자 권한으로 접속**
**1-2. 다음의 내욕을 복사 붙여넣기(각각 복사 후 우클릭)**

:: Update of PIP  
pip install --upgrade pip  
python -m pip install --user --upgrade pip  

:: Jupyter Nbextensions  
pip install jupyter_contrib_nbextensions  
jupyter contrib nbextension install --user  

:: Jupyter Lab  
pip install jupyterlab  
pip install --upgrade jupyterlab  

:: Jupyter Lab Extensions Package  
pip install nodejs  
conda install --yes nodejs  
conda install -c conda-forge --yes nodejs  

:: Table of Contents  
jupyter labextension install @jupyterlab/toc  

:: Shortcut UI  
jupyter labextension install @jupyterlab/shortcutui  

:: Variable Inspector  
jupyter labextension install @lckr/jupyterlab_variableinspector  

:: Go to Definition of Module  
jupyter labextension install @krassowski/jupyterlab_go_to_definition  

:: Interactive Visualization  
jupyter labextension install @jupyter-widgets/jupyterlab-manager  
jupyter labextension install lineup_widget  

:: Connection to Github  
jupyter labextension install @jupyterlab/github  

:: CPU+RAM Monitor  
pip install nbresuse  
jupyter labextension install jupyterlab-topbar-extension jupyterlab-system-monitor  

:: File Tree Viewer  
jupyter labextension install jupyterlab_filetree  

:: Download Folder as Zip File  
conda install --yes jupyter-archive  
jupyter lab build  
jupyter labextension update --all  

:: End  

## 2. Jupyter Notebook 확장 사용

**2-1. Jupyter Notebook의 경로 설정변경('C:' -> 'D:')**
- Jupyter Notebook 속성 클릭
- '대상'에서 '%USERPROFILE%/'를 'D:'로 변경
- '시작위치'에서 '%HOMEPATH%'를 'D:'로 변경 후 확인

**2-2. Jupyter Notebook의 기능 추가**
- Jupyter Notebook 상단에 Nbextensions 클릭(1번의 과정을 진행한 후에 업로드)
- 'diable configuration for nbextensions without explicit compatibility'를 체크 혹은 체크 풀기
- 아래의 7개의 기능을 확인 후 추가, 더불어 NBextensions의 내용을 확인하고 필요하거나 도움이 될만한 것들을 활용하도록 한다.
  - Table of Contents
  - Autopep8
  - Codefolding
  - Hide Input All
  - Execute Time
  - Variable Inspector

**2-3. Jupyter Lab 사용**
- 상단의 URL을 'http://localhost:8888/lab' 로 설정
- 좌측 상단의 폴더를 활용한 다양한 구현기능 가능(Pycharm에서 활용하는 방법과 비슷)
