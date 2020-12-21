# korean_sentiment_analysis
KoBERT를 사용한 한국어 감정 분석 

## 설명
BERT는 Transformer의 인코더만을 이용해 양방향 사전학습을 진행한 언어 모델이다.
본 프로젝트에서는 한국어 Bert 모델인 KoBERT를 사용하여 네이버 영화리뷰에 대한 감정 분석을 진행한다. 
KoBERT는 SKTBrain에서 한국어 BERT 모델을 개선하기 위해 개발한 한국어 전용 BERT로 위키피디아와 뉴스 데이터를 기반으로 25백만 문장을 학습하였고 8,002개의 단어 리스트를 보유하고 있다.

## Installation
- Python 3.7
- PyTorch 1.7.0
- transformers 4.1.1
- MXNet 1.7.0
- gluonnlp 0.10.0
- onnxruntime 1.6.0
- sentencepiece 
- kobert
- sklearn
- 

## Input data
Naver sentiment movie corpus v1.0 : https://github.com/e9t/nsmc
- ratings_train.txt: 150K reviews
- header : id, document, label (0,1)
- 80% Train, 20% Validation data로 분할하여 학습 진행

## 환경
- colab 개발환경 사용
- gpu 사용으로 학습속도 향상

## Training 실행
1. Import Modules
 - Installation의 모듈들을 설치한다.
2. Data Load
 - 올바른 train data의 Path를 입력하여 train data를 로딩한다.
 - sklearn.model_selection.train_test_split을 활용해 train data를 8:2의 비율로 train과 validation data로 분할한다.
 - Data의 label 분포를 통해 Imbalance 여부를 확인한다.
 - Data의 document의 최대, 평균 길이를 확인한다. 이후 단어 임베딩 시 max_length를 결정할 때 사용한다.
 
3. Preprocessing
 - KoBERT tokenizer를 사용해 data를 tokenize 한다.
 - gluonnlp.data.BERTSentenceTransform을 사용해 BERT 모델에 적합하게 단어 임베딩한다. Document의 평균 길이는 35로, max_sequence_length = 40으로 설정하며, Padding하여 문장 길이를 매칭한다.
 - Data의 모든 문장들에 대하여 임베딩을 진행한다.
 - 임베딩된 문장에서 padding이 아닌 부분은 1, padding 부분은 0으로 입력하는 attention mask를 만들어 0인 부분은 attention을 수행하지 않도록 한다.
 
4. Modeling
 - torch.utils.data.dataloader를 사용해 문장 입력, attention mask, label 데이터를 묶고 설정한 배치사이즈 만큼 데이터를 가져온다.
 - 배치사이즈는 16으로 설정한다.
 - Training 모델은 Kobert 모델을 사용한다.
 - Linear classifier로 Label 분류를 수행한다. 

5. Training
 - gpu 사용한다. (cuda library)
 - Optimizer : AdamW
 - Loss 함수 : Cross entropy
 - Epoch : 4, Learning rate : 5e-5
 - Train 데이터로 학습하고 validation 데이터로 검증한다. 매 Epoch마다 loss, accuracy를 계산하여 학습을 검증한다.

## Test 실행
- Train 데이터와 동일한 폴더에 Test할 데이터를 저장한다.
- 단어임베딩과 attention mask를 만들고 학습 모델에 입력한다.
- 출력값을 실제 데이터 label과 비교하여 모델의 성능을 테스트한다. 

## Reference
KoBERT : https://github.com/SKTBrain/KoBERT.git
https://github.com/deepseasw/bert-naver-movie-review.git
https://mccormickml.com/2019/07/22/BERT-fine-tuning/
