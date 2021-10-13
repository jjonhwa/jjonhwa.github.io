---
layout: post
title: "Second P-stage 6(Day 42)"
categories: boostcamp
tags: main
comments: true
---
GPT(Generation Pre-trained Transformer), KoGPT 활용법에 대하여 학습한다.

**부스트 캠프 42일차 학습 요약**
- **학습** : GPT 언어모델, GPT 언어 모델 기반의 자연어 생성
- **P-Stage** : NER Pos tagging, Data Augmenatation
- **피어 세션**

## 목차
- [1. GPT](#1-gpt)
- [2. GPT-2](#2-gpt-2)
- [3. GPT-3](#3-gpt-3)
- [4. KoGPT-2를 활용한 자연어 생성](#4-kogpt-2를-활용한-자연어-생성)

## 1. GPT
### 1-1. BERT vs GPT
- BERT는 자연어에 대한 embedding 언어 모델
- GPT는 자연어의 생성에 특화된 언어 모델

### 1-2. GPT 모델 소개
![31](https://user-images.githubusercontent.com/53552847/137043229-497cb1c0-74e7-48a7-893d-cf64baebf539.PNG)
- BERT는 Transformer의 Encoder를 사용했으며, **GPT는 Trnasofrmer의 Decoder를 사용한 모델**이다.
![32](https://user-images.githubusercontent.com/53552847/137043230-9a082c1f-d142-4d45-9af2-f95c5d8de745.PNG)
- 위의 그림에서 볼 수 있듯이, **GPT는 주어진 단어에 대하여 다음에 올 단어가 무엇인지를 학습**한다.
![33](https://user-images.githubusercontent.com/53552847/137043232-2199bb3b-a100-43fc-b949-2e2077415f23.PNG)
- GPT-1의 경우, BERT보다 먼저 등장하였으며, 목적은 RNN과 같이 기존의 문장이 들어왔을 때 그 문장에 대한 **context vector를 출력**하고 이에 **linear layer를 추가하여 분류 task를 적용**하기 위해 설계된 모델이다.
- 위의 그림에서 볼 수 있듯이, GPT-1은 classification, similarity 계산 등의 task에 적용하는 모델이다.

### 1-3. GPT 특성
- [자연어 문장 -> 분류] 성능이 아주 좋은 디코더
- 적은 양의 데이터에서도 높은 분류 성능을 나타낸다.
- 다양한 자연어 task에서 SOTA 달성 (BERT 이전)
- Pre-train 언어 모델의 새로운 지평을 열었다. -> BERT 발전의 밑거름
- 하지만, 여전히 지도 학습을 필요로 하며(fine tuning), labeled data가 필수적
- **특정 task를 위해 fine-tuning된 모델은 다른 task에서 사용이 불가능하다**라는 단점이 있다.

### 1-4. GPT 발전과정
- 위의 GPT 특성으로부터, GPT는 아래에서 설명하는 방향으로 발전하였다.
- '특정 task를 위해 fine-tuning된 모델은 다른 task에서의 사용이 불가능하다'라는 점으로 부터 생각해보았을 때, 언어의 특성상 지도학습의 목적함수와 비지도 학습의 목적함수는 동일하므로 fine-tuning 할 때 목적함수와 pre-training 할 때의 목적하뭇가 같으므로 **fine-tuning의 필요성이 없지 않을까**라는 생각으로 이어졌다.
- 엄청 큰 데이터셋을 사용하면 자연어 task들을 자연스럽게 학습한다. 이를 바탕으로 지도학습, 비지도 학습을 구분할 필요가 없어지고, fine-tuning 역시 할 필요가 없을 것이다. 라는 가설을 가정으로 둔다.
- 인간은 새로운 task를 학습하는데 있어 수 많은 데이터를 필요로 하지 않으므로, 기계 역시 pre-trained model로 부터 fine-tuning하여 하나의 모델이 하나의 task만 수행하는 것은 옳지 않다라고 판단한다.

### 1-5. fine-tuning vs few-shot learning
- Fine-tuning은 pre-trained model에 나의 dataset을 넣어가면서 gradient update를 진행하고 하나의 task에 점점 알맞게 만드는 것이다.
- zero-shot, one-shot, few-shot learning
    - Gradient update가 존재하지 않는다.
    - inference과정에서, 내가 원하는 task에 대한 hint를 몇 개를 주냐에 따라서 zero-shot, one-shot, few-shot으로 나뉜다.
    - 이 아이디어를 바탕으로 GPT-2가 개발되었다.

## 2. GPT-2
![34](https://user-images.githubusercontent.com/53552847/137043233-0f010934-401a-48f5-9119-4a2325504888.PNG)
- GPT-2의 경우, Dataset의 양을 불리고, HyperParameter의 개수도 약 10배정도 늘려서 학습을 진행했다.
- GPT-2를 개발한 후, zero-shot, one-shot, few-shot Learning이 성공적으로 이루어졌음을 증명했다.

### 2-1. GPT-2 구조
![35](https://user-images.githubusercontent.com/53552847/137043234-5c581fc4-528b-4d90-8e65-0007485c1d8f.PNG)
- GPT-1의 Decoder에서 구조만 조금 다르게 구성되어 있다.
- 실제로 GPT-2를 이용했을 때, 다음 단어 예측하는 방식으로 SOTA 성능이 나타났다.
- zero-shot, one-shot, few-shot을 통해서, 기계 독해, 요약, 번역 등의 자연어 task를 수행했을 때, 일반적으로 학습되어있는 신경망과 비슷한 수준까지 올라왔다.
- zero-shot, one-shot, few-shot learning이라는 개념의 새로운 지평을 제시했다.

## 3. GPT-3
![36](https://user-images.githubusercontent.com/53552847/137043235-8bee2ad5-2c93-4f89-9ab0-b0f7c235b74d.PNG)
- 기존 HyperParameter(GPT-2)를 약 100배 늘렸다.
- 사이즈도 늘리고 학습데이터 역시 100배 이상 늘렸다.

![37](https://user-images.githubusercontent.com/53552847/137043237-8ae1c9a7-7341-48aa-bc89-7ffee5884b5b.PNG)
- GPT-3 역시 Transformer Decoder를 사용했다.
- GPT-2와 동일한 decoder를 사용한 것은 아니고, 위의 그림처럼 약간 다른 구조와 initialize를 수정한 구조를 가지고 있다. 전체적인 구조는 transformer decoder를 사용한 것으로 동일하다.
![38](https://user-images.githubusercontent.com/53552847/137043238-04f4e0a4-f024-41d1-a567-037449a0001f.PNG)
- 위의 표에서 처럼, Control은 사람이 직접 쓴 뉴스기사이고, GPT-3 175B는 GPT-3가 만든 뉴스 기사인데, 이를 어느정도 사람이 썻는지를 비교했을 때, Control의 경우 88%, GPT-3의 경우 52%를 사람이 쓴 거라고 판단할 만큼 유의미하게 작성하였따.
![39](https://user-images.githubusercontent.com/53552847/137043239-87beba97-6752-43b4-aa11-c17584fd06b6.PNG)
- GPT-3를 활용한 사칙연산 실험을 진행했을 때, 두 개의 연산에 대해서는 거의 100% 정확도에 달했다.

### 3-1. Open Domain QA
- 기계 독해의 경우, 정답에 관련된 context가 input으로 함께 주어지는데, Open Domain QA의 경우 이러한 문서가 주어지지 않고 질문이 주어져 이에 대한 올바른 답변을 하는 task이다.
- ODQA의 경우 굉장히 어려운 task이고, 모델 자체가 상식적인 것과 어려운 지식들을 모두 가지고 있어야 답변이 가능하다.
- GPT-3의 Few-shot learning을 사용했을 때, 아래의 표와 같은 성능이 나왔고, 다른 모델 대비 SOTA 성능을 나타냈다.
![40](https://user-images.githubusercontent.com/53552847/137043214-3ca0c355-65a5-497f-8bdc-460603f1e872.PNG)

### 3-2. GPT-3의 ODQA 예시
![41](https://user-images.githubusercontent.com/53552847/137043217-877323ca-d1b7-43ba-ade7-3131f6c750df.PNG)
![42](https://user-images.githubusercontent.com/53552847/137043220-8efc9b3a-2c33-43de-a34a-4480e4c10857.PNG)
![43](https://user-images.githubusercontent.com/53552847/137043221-4883d0cb-0e43-4dda-be04-db9cde25907e.PNG)
- GPT-3를 활용한 예시는 위의 이미지들과 같다.
- Awesome GPT-3(OpenAI GPT-3 API)에서, GPT-3를 활용한 70개가 넘는 다양한 어플리케이션 예제를 확인할 수 있다. 

### 3-3. GPT-3의 한계
- 다음 단어 혹은 asked 단어 예측하는 언어 모델 학습 방식으로 정말 모두가 해결될까?
- Weight update(gradient update)가 없다는 것은 모델에 새로운 지식 학습이 없다는 것을 의미하며, 이는 시기가 달라지는 문제에 대응하기 힘들다라는 것을 의미한다.
- 모델 사이즈만 키우면 되는가? -> 다른 연구 방향 필요(윤리적 문제, 시대에 따른 언어 표현방식의변화 등)
- Multi Modal 정보 필요 -> GPT는 오직 글로만 세상을 학습한다.

## 4. KoGPT-2를 활용한 자연어 생성
### 4-1. KoGPT-2 활용법
- Tokenizer 로드
- KoGPT2의 경우 vocab size가 50000으로 되어있으므로, GPT2Config를 활용하여 **config를 가져올 때 반드시 vocab_size를 맞춰주어야 한다.**
- pad_token_id에 대한 값을 명시적으로 알려준다.
    -  config.pad_token_id = tokenizer.token_to_id('<pad>')와 같다.
- 모델 역시 저장된 directory로부터 불러오고, GPT2LMHeadModel 패키지로부터 불러올 수 있다.
- 모델을 로드할 때, map_location option을 'cuda'로 설정해야지 GPU로 올려서 사용할 수 있다.
- Grid Search
    - model.generate를 사용한다.
    - 아무 option도 주지않고, `input_ids`와 `max_length`만 줄 경우 가장 최고 성능만 추출하는 grid_search 방법으로 generate가 진행된다.
    - 같은 단어에 대한 다음 단어 역시 무조건 똑같이 생성된다.
- Beam Search
    - `num_beams`를 지정해줌으로서 상위 num_beams의 개수만큼의 문장을 선택하고 그 중 가장 높은 확률을 가지는 문장을 출력한다.
    - 여러 가지의 문장들의 확률을 파악하면서 진행해야 하기 때문에 Inference 시간이 오래걸린다.
    - 단순하게 num_beams만 활용할 경우, 반복적인 문장이 생성될 수 있다.
        - 반복적인 문장 생성을 막기 위해, n_gram 패널티를 줄 수 있다.
        - n_gram 패널티는 특정한 음절들이 n번 이상 반복적으로 등장할 경우 패널티를 주는 것이다.
        - `no_repreat_ngram_siz`e option를 통해서 패널티를 줄 수 있다.
        - 하지만, n_gram 패널티의 경우 신중히 사용되어야 한다. 고유 명사의 경우 자주 등장할 수밖에 없는데 n_gram 패널티를 줄 경우, 지정 개수밖에 등장할 수 없기 때문이다.
    - `num_return_sequences` 옵션을 통해 최종 몇 개의 문장을 반환할 것인지 정할 수 있따.
        - Beam Search를 사용한다고 하더라도, 약간만 다를뿐이지, 결국 비슷한 결과물이 출력된다.

### 4-2. KoGPT-2 옵션 활용
- 위에서 설명한 model.generate의 option으로서 `input_ids`, `max_length`, `num_beams`, `no_repeat_ngram_size`, `num_return_sequences` 등을 살펴보았다.
- 추가적으로 좀 더 유연한 결과물을 만들기 위해서 활용할 option들에 대하여 살펴보도록 한다.

#### Human vs BeamSearch
![44](https://user-images.githubusercontent.com/53552847/137043222-751fb5e9-1afd-422c-8ef2-cc1a333d9199.PNG)
- Beam Search를 통한 문장의 완성도는 거의 100에 가깝다.
- 하지만, 사람이 만들어낸 text를 확률값으로 계산했을 때, 반드시 높은 수준으로만 유지되는 것은 아니고 낮은 수준의 확률을 가진 문장, 높은 확률을 가지는 문장들이 랜덤하게 분포하고 있음을 볼 수 있다.
- 완벽한 문장을 생성하게 되면, 오히려 똑같은 문장만 반복적으로 생성하게 된다.
- 약간의 노이즈를 섞어야지 더 자연스럽고 다이나믹한 문장이 나올 수 있음을 어느 정도 알 수 있다.

#### Sampling
- 조건부확률 값에 따라서, 단어를 무작위로 선택하는 방법
- `do_sample option`을 True로 주면 Random Sampling이 시작된다.
- `top_k` 내에서 랜덤 샘플링을 진행하게 되는데, 이 때 `top_k` option을 0으로 주게되면, 완전한 랜덤 샘플링이 진행된다.
- 이러한 random sampling option을 추가함으로서 문장이 일관성있고 다이나믹하게 나올 수 있도록 조절할 수 있다.
- Sampling with Softmax (temperature를 낮춰 분포를 더 선명하게 만드는 방법)
    - 높은 확률의 단어가 나올 확률을 더 높게, 낮은 확률의 단어가 나올 확률을 더 낮게 단어들이 뽑히도록 샘플링하는 기법이다.
- Top-K Sampling
    - 높은 확률을 가진 k개의 단어들을 선정하고 여기서 랜덤으로 샘플링하는 기법이다.
- Top-p Sampling
    - 높은 확률을 가진 단어부터 누적확률이 p가 될 때까지의 단어들만 추출하여 이들 중 랜덤으로 선택하는 기법이다.
- 실제로 Top-p, Top-k 두 방법 모두 잘 작동한다고 한다.
- `eos_token_id`를 입력하고, `early_stopping` option을 True로 주게되면, 지정한 eos_token_id가 등장할 때 문장 생성을 종료한다.
- `bad_words_ids`를 명시해주면, bad_words_ids가 등장할 때, 다른 단어를 선택하도록 지정할 수 있다.
