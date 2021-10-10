---
layout: post
title: "정규표현식 - 메타문자"
categories: booststudy
tags: plus
comments: true
---

---
layout: single
title: "KLUE Relation Extraction: Subject Entity and Object Entity"


---

**🥇 AUPRC score of 83.2 ranked the 1st place / 19 teams**

**🥉 F1 score of 73.9 ranked 3rd place / 19 teams**

[KLUE(Korean Language Understanding Evaluation) Benchmark](https://paperswithcode.com/dataset/klue) is newly introduced Korean NLP Benchmark. In Naver Boostcamp, the team solved Relation Extraction task from KLUE Benchmark dataset within the span of 12 days.

## Dataset & Dataloader

### Entity Representation

**Putting labels(or entity markers) around the target entity has boosted the performance.**

![image-20211008223302060](../assets/images/2021-10-08-Relation-Extraction/image-20211008223302060.png)

How to annotate entity marker around the target entities was important point of discussion.The paper [An Improved Baseline for Sentence-level Relation Extraction(Zhou et al. 2021)](#https://arxiv.org/pdf/2102.01373v3.pdf) gives 5 types of composite entity information labelling technique as depicted above.

Typed entity marker method was good but required two additional steps. It required adding special token to the tokenizer and resizing the embedding layer accordingly. 

Even though it is good idea, this has few shortcomings. **Creating and importing new tokens with no pretrained hidden features might hinder the model's performance.** **Moreover, additional part of speech(POS) tagging information is not going in as the input to the pretrained model, but only regarded as special tokens.**

With such background, the paper Improved Baseline proposes "Typed entity marker punctioation method". **Enclosing the entity with entity span(POS) and entity types(SUB/OBJ) without introducing new special tokens**. This paper argues that enclosing the subject and object entities with “@” and “#” alone boosts the perfomance. Performance comparisions are given with the table below. 

The team utilized [Pororo NER POS Tagging function](#https://kakaobrain.github.io/pororo/tagging/ner.html) for both train and test dataset. With Pororo POS tagger, the team experimented both (1) typed entity marker and (2) typed entity punctutation method. (1) Typed entity marked was fed into as an input of RBERT custom model, whereas (2) typed entity punctuation method was fed into its own custom model.

### (1) Typed Entity Marker Dataset Creation

Instead of saving the tokenizer itself, the special tokens added with Pororo NER POS Tagging were saved as txt format file. It was later loaded as txt file, added to tokenizer with add_special_tokens() method. The embedding layer was resized accordingly. However, it is still remain unsolved that whether resizing embedding layer boosts the performance or harms the performance.

The following CSV files were created for time-saving purpose: Pororo POS tagging took 12+ hours for 40k train + test sentences.

  - [Pororo POS Tagging with [SUB-{POS}] [/SUB-{POS}] entity special tokens](#https://raw.githubusercontent.com/boostcampaitech2/klue-level2-nlp-15/main/dataset/train_pororo_sub.csv?token=AG3HZNZI4TIT7XLSBIMJYHDBL4EVE)
  - [Pororo POS Tagging with improved baseline style `#^organization^대안신당#`](#https://raw.githubusercontent.com/boostcampaitech2/klue-level2-nlp-15/main/dataset/train_typed_entity_marker_punct.csv?token=AG3HZN4TYFCN3HXLUWYQ7Y3BL4EVG)
  - [Pororo POS Tagging with improved baseline style translated into Korean `#^사람^조지 해리슨#`](#https://raw.githubusercontent.com/boostcampaitech2/klue-level2-nlp-15/main/dataset/train_punct_kor.csv?token=AG3HZN3VUMQMEUL337RH333BL4EVC)

### (2) Typed Entity Marker Punctuation Dataset Creation

|                                                   |                           Sentence                           |
| :-----------------------------------------------: | :----------------------------------------------------------: |
|    **Typed Entity Marker Punctuation Method**     | **#^ORG^조지 해리슨#이 쓰고 @\*ORG\*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.** |
| **Typed Entity Marker Korean Punctuation Method** | **&^단체^조지 해리슨&이 쓰고 @\*단체\*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.** |

Rather than merely implementing the paper [An Improved Baseline for Sentence-level Relation Extraction(Zhou et al. 2021)](#https://arxiv.org/pdf/2102.01373v3.pdf), the team went two steps further. 

First, we translated the english entity marker punctuation into Korean word. This was because pretraind `klue/roberta-large` tokenizer's english vocabulary sets are more or less than 800 words out of 32000 token vocab sets. Thus the team decided that using Korean punctuation rather than english punctuation in order to boost performance.

Moreover, the team replaced `#` punctuation representation with `&` representation. Since the team is using pretrained RoBERTa, `#` is often found in tokenizers as sub-word representations(`안녕하세요` -> [`안녕`, `##하세요`]). Therefore we thought that `&` would be better for annotation for the punctuation representation.

### Setting maximum token length for RoBERTa tokenizer

[📈 wandb: Comparison through changing "max_length" parameter](#https://wandb.ai/danielkim30433/huggingface/reports/Comparison-through-changing-max_length-parameter--VmlldzoxMDc1NTE0?accessToken=4r9exwlxam05iy0u0aaexwxfpgl041hyno7gsxthwygf37hgq4xg2kbj66dtj1gf) 

[🐍 Jupyter Notebook for EDA for KLUE Relation Extraction dataset](#https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/dev/v03/token_length%EA%B4%80%EB%A0%A8.ipynb)

There are only 460 sentences out of 32.5k train set that exceeded the token length of 128, using `klue/roberta-base` wordpiece tokenizers. Thus we set the maximum token length as 128. 

Thanks for my teammate's concise inspection for finding optimal max_token_length in EDA, **it is confirmed that each of the class label's token length distribution are uniform to one another**. Therefore even if we drop 460 sentences which exceeds maximum_token_length, balance of data between the classes wouldn't be affected.

|                           Class 1                            |                           Class 2                            |                           Class 3                            | ...  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--: |
| ![img](../assets/images/2021-10-08-Relation-Extraction/7E22FA0E-BBEC-4742-96C8-46E316D30CCA.png) | ![img](../assets/images/2021-10-08-Relation-Extraction/AE0C1743-6B50-4819-898D-D0500A671AEE.png) | ![img](../assets/images/2021-10-08-Relation-Extraction/F4C5C509-3CA8-4FEB-8E91-A00F2E1BD7BA.png) |      |

It was important to summate token length +4 in order to compensate for later added special entity tokens. For example, four entity tokens added to the sentence were `[SUB-ORGANIZATION]`, `[/SUB-ORGANIZATION]`,  `[OBJ-LOCATION]`, `[/OBJ-LOCATION]`. 

It should be careful that padding="max_length", not set as padding=True, so that the sentences are truncated(or sliced) properly.

![image-20211008012353902](../assets/images/2021-10-08-Relation-Extraction/image-20211008012353902.png)

![image-20211008012403171](../assets/images/2021-10-08-Relation-Extraction/image-20211008012403171.png)

Batching strategies by [using huggingface's data collate class](#https://huggingface.co/transformers/main_classes/data_collator.html), [applying dynamic padding](#https://youtu.be/7q5NyFT8REg), or making custom [uniform length batching function](#https://mccormickml.com/2020/07/29/smart-batching-tutorial/) is known to decrease the train time significantly as following. ![image-20211008012609016](../assets/images/2021-10-08-Relation-Extraction/image-20211008012609016.png)

It is noteworthy that uniform length batching(or smart batching on the table) increased the model's performance. Due to time constraint, in this competition we applied dynamic padding only.

batch_size were selected to be bigger than 30 since we assumed batch numbers being bigger than class numbers would have reduced the bias towards certain classes.

### Data Augmentation

Attempted KoEDA(Easy Data Augmentation implemented in Korean Language) for the dataset. Teammate applied Random Insertion, Random Deletion, Random Swap, Synonym Replacement but this have not led to notable increase in performance.

## Model & Finetuning

### How to select pretrained model

We fixed error entailed from using klue/roberta-large huggingface model [and provided walk-through solution on this Github Issue](#https://github.com/KLUE-benchmark/KLUE/issues/33). This was fixed by setting `return_token_type_ids=False` argument on the tokenizer encoding function. Since RoBERTa didn't have token type mask, tokenizer should not return token_type_ids.

  ```python
  tokenized_sentences = tokenizer(
          concat_entity,
          list(dataset["sentence"]),
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=256,
          add_special_tokens=True, 
          token_type_ids = False )# for klue/roberta-large
  ```

Pretrained model selection was based on given model benchmarks in fields of task. The task we focused on were relation extraction. If not given the corresponding performance for RE(relation extraction), we looked at Named Entity Recognition and Natural Language Inference.

Model Benchmarks given by [KLUE's benchmark scores](#https://github.com/KLUE-benchmark/KLUE#baseline-scores)
We also referred to another model benchmarks given by [Tunib-Electra's benchmark scores](https://github.com/tunib-ai/tunib-electra) and [KoElectra's benchmark scores](https://github.com/monologg/KoELECTRA)

We first tested KoElectra as backbone model, since its NER performance was the highest. But we later on switched to RoBERTA.

[By counting the number of unk tokens of the pretrained models](#https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/snoop/notebooks/EDA.ipynb), we assumed that the lower unknown tokens appeared, the better the model's performance it would be. **Given that the number of unknown tokens are the fewest on klue/roberta-base model, we decided to stick with the klue-pretrained models.**

### How we Fine-tuned the model according to the relation extraction task

[Based on TACRED Benchmark Leaderboard](#https://paperswithcode.com/sota/relation-extraction-on-tacred), our team decided to test SOTA model according to the order. Along with TACRED score, the team also referred to [Matching the Blanks: Distributional Similarity for Relation Learning. Soares et al. 2019](#https://aclanthology.org/P19-1279.pdf) to find where to extract hidden features from.

According to [Matching the Blanks: Distributional Similarity for Relation Learning. Soares et al. 2019](#https://aclanthology.org/P19-1279.pdf), The best tokens to extract features from were either (1) pooling multiple entity tokens' hidden states or (2) extract the first entity markers of each subject entity and object entity. Provided model's structure from the paper are given as below, along with its methods' performance.

![image-20211010175325279](../assets/images/2021-10-08-Relation-Extraction/image-20211010175325279.png)

![image-20211010175418328](../assets/images/2021-10-08-Relation-Extraction/image-20211010175418328.png)

Thus the team chose customized Improved Baseline model for `(f) Entity Markers -Entity First` method and RBERT for `(e) Entity Markers - Mention Pool` method.

**[🤖 Customized Improved Baseline Custom Model](https://paperswithcode.com/paper/an-improved-baseline-for-sentence-level) (f1 0.736)**
  ![image-20211008225119287](../assets/images/2021-10-08-Relation-Extraction/image-20211008225119287.png)

  - [(1) first version code @dev branch](https://github.com/boostcampaitech2/klue-level2-nlp-15/tree/dev)
  - [(2) Improved_Baseline code](https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/main/RE_improved_baseline.ipynb) (f1 0.732)
  - [(3) Improved Baseline kfold code]() (f1 0.736)
  - [Notes for Improved Baseline](https://global-sunset-142.notion.site/RE_improved_baseline-a6f4bb9ecd2c4362996d18df507f4327)

**[🤖 Customized RBERT](https://paperswithcode.com/paper/enriching-pre-trained-language-model-with) (f1 0.745, [📈 wandb](https://wandb.ai/snoop2head/KLUE-RE?workspace=user-snoop2head))** 

👉 [Please refer to my blog post for detailed RBERT finetuning process](https://snoop2head.github.io/Relation-Extraction-Code/)
  ![image-20211008225126824](../assets/images/2021-10-08-Relation-Extraction/image-20211008225126824.png)

**[🤖 Customized Concat Model](#https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/085dd842e93a8d1e0d30f8cf42268ddb96cafd4a/notebooks/train_with_pororo_with_val_klue-roberta.ipynb) (f1 0.730, [📈 wandb](#https://wandb.ai/snoop2head/huggingface/runs/1d3hlfn6?workspace=user-snoop2head))**

The concat method was given from the boostcamp as a baseline. It was simply putting Entity 1 and Entity 2 with `[SEP]` separator in the beginning of the tokenized input. We tried to exclude the concat in the beginning of the tokenized sentence, but this led to worse evaluation loss and evaluation accuracy: [📈 wandb](https://wandb.ai/danielkim30433/huggingface/runs/3e1lomf5?workspace=user-danielkim30433). We concluded that putting concated subject entity and object entity in the beginning boosts the classification. Thus we also adopted customized concat model.

- [(1) pororo POS & Entity tagged & stratified 5-fold cross validation ](#https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/main/train_with_pororo.ipynb)
- [(2) replaced bert model with xlm-roberta & applied focal loss(gamma=0.5)](#https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/snoop/notebooks/train_with_pororo_with_val.ipynb) ([📈 wandb](#https://wandb.ai/snoop2head/huggingface/runs/2576sujh), f1 0.70)
- [(3) used klue/roberta-large model & applied focal loss(gamma=0.5)](#https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/snoop/notebooks/train_with_pororo_with_val_klue-roberta.ipynb) ([📈 wandb](#https://wandb.ai/snoop2head/huggingface/runs/1d3hlfn6?workspace=user-snoop2head), f1 0.733)

**[🤖 RECENT / Duo classifier](#https://github.com/Saintfe/RECENT) (f1 0.60)** required more time to study for our team and has not been fully implemented.

### Findings from fine-tuning methods

Number of Epochs were decided by monitoring model's performance with 📈 wandb. KPI for the model's performance was set as evaluation loss. 

- It was evident that concat model should not exceed more than 2000 training steps(or 3 epochs) as evaluation loss increased after then. 
- However, RBERT's evaluation loss kept decreasing until 3200 training steps(or 5 epochs). This might possibly show that **RBERT providing more information about the entities' relation within the sentence with additional masks increases robustness.**
- Improved Baseline is light-weight. Compared to RBERT which takes up 26GB of GPU capacity and taking 10hrs for 5 fold training, Improved Baseline on the other hand takes up about 15GB of GPU capacity and takes more or less than 4 hours for training. 
- Let's compare RBERT and Improved Baseline method. It is surprising that light-weight Improved Baseline(f1: 0.736) almost yielded similar performance as RBERT(f1: 0.745). 
- Improved Baseline only extracts first entity token marker whereas RBERT averages all the tokens between the entity tokens. **One of the downside of averaging the hidden features of multiple entity tokens is that it ignores the order of the tokens given in the sentence.** I think this is the reason that extracting the first token of the entity resulted in similar f1 score with RBERT method.

## Training

### Why we selected FocalLoss as the criterion

![img](#https://user-images.githubusercontent.com/30318926/134863763-eac345a4-b4c2-46f3-b093-1fcbb284ce70.png)

Looking at such class imbalance, the team tried to apply the distrubution of such classes as weights to CrossEntropy criterion and to LabelSmoothing criterion. It was worth trying, but it was better to apply focal loss, which yielded better output for the previous p-stage competition, as the criterion.
![image-20211008014220660](../assets/images/2021-10-08-Relation-Extraction/image-20211008014220660.png)

### How we selected the gamma parameter for Focal Loss

The team experiment gamma paremeter by monitoring wandb for first 500 steps. For the first 500 steps, we designated f1 score and evaluation loss as the key primary index to determine the criterion's performance.

- FocalLoss 0.5, AdamP (lowest val/loss) -> 500 step, eval loss 0.69 is considered to be higher than the concat model's evaluation loss so the team increased the gamma value. 
- FocalLoss 1.0, AdamP (lowest val/loss) -> 500 step, eval loss 0.56 is considered to be adequate which was lower than concat model's loss. I was aware of the fact that increasing gamma value might risk the model to fall within the local minimum, so I stopped at 1.0 
- Teammate experimented gamma parameter started off from 2.0, which is concluded to be inadequate, thus the team's gamma parameter converged to 1.0.

### How we selected the optimizer AdamP vs AdamW

Custom model's results were compared using wandb for first 500 steps. We concluded that AdamP of ClovaAI allowed model to converge the loss slightly faster than AdamW provided by huggingface. This is in accordance with ClovaAI team's benchmark when compared AdamW and AdamP.

![img](#https://clovaai.github.io/AdamP/static/img/table07.svg)

### Other important details

- Applying Stratified KFold alone increased the model's performance by +3%p for f1 score (0.67 -> 0.7) for public leaderboard
- Schedule: linear vs cosine
- Saving best model strategies were differed: accuracy, f1 score, eval/loss. 

### Inferencing

- Out of fold ensemble when each of the is trained on train dataset.
- Soft-voting ensemble the 1) best concat model, 2) best improved baseline model, 3) best RBERT model. Each of the model's saving criteria was different, which enabled further performance boost when scored in private leaderboard.
  1. Best concat model was based on best accuracy
  2. Best Improved Baseline model was based on best f1 score
  3. Best RBERT model was based on lowest evaluation loss

---

### Other details to try Next time

- [ ] str.replace() sometimes replaces not only target words/tokens, but other words/tokens that has same characters. If given the index of the target entity/token, it is better to use that even if it is such a nuisance.
- [ ] Task Adaptive Pretraining(TAPT): Pretraining for the given dataset was refrained for this competition. Since Klue/roberta models were already pretrained on task dataset, we saved time not doing pretraining. However, when given with specific domains such as math, law, science, I think it is still worth trying MLM tasks and NSP tasks based pretraining. TAPT was allowed for both train and test datasets for this competition
- [ ] Uniform Length Batching using huggingface's collate class / or making on my own.
- [ ] optuna fine-tuning
- [ ] Round-trip translation / Back translation based on cosine similarity for data augmentation
- [ ] Adding bidirectional GRU(or LSTM) layers after the transformers layer for the sake of non-linearity.
  - [ ] [had difficulty of certain proportion of validation dataset being dropped]()
  - [ ] had difficulty of matching the torch tensor size in between transformers encoder layer to fully connected classifier layer.

### Teamwork 👨‍💻

- It was our team's first time to fine-tune BERT model for the Relation Extraction task. We set our goals to the absolute value of f1 score(0.75) and never gave up. I never looked at the leaderboard until the end of competition in order to enjoy the process of learning.
- Purpose of participating competition was to familiarize ourselves to fine-tuning BERT model. Finding the effective way to preprocess/train/infer was our objective. 
- Thus our team spent most of the time doing custom preprocessing and custom model fine-tuning method of our own. Along with increasing the f1 score, it was our goal to practice writing code that reflects our idea.
- Since group of 7 people had to work as a team remotely, we set our ground rule not to cut in while other person is speaking. 
- Our team decided to respect each teammate's beliefs. If teammate wants to commence certain experiment and believes that it will contribute to robustness and performance of model, we promised to patiently wait and help the experiment. 
- By sharing both of Jupyter notebook links and wandb links, it helped other teammates to utilize(or copy & paste) code of the teammate and observe the performance.
- While running the model and GPU is occupied, our teammates helps to debug other teammate's code. For example, if one GPU is occpupied with MLM TAPT, then the teammate "live coded" putting bidirectional gru between the classifier and last transformer encoding layer.

### Reference

- [KLUE RE Relation Extraction Dataset / CC BY-SA](#https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark)
- [An Improved Baseline for Sentence-level Relation Extraction(Zhou et al. 2021)](#https://arxiv.org/pdf/2102.01373v3.pdf)
- [Matching the Blanks: Distributional Similarity for Relation Learning. Soares et al. 2019](#https://aclanthology.org/P19-1279.pdf)
- [Enriching Pre-trained Language Model with Entity Information for Relation Classification, Wu et al. 2019](#https://arxiv.org/pdf/1905.08284v1.pdf)
