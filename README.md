# OvercomingPositionEmbeddingLimitation

본 저장소는 '사전 학습 모델의 위치 임베딩 길이 제한 문제를 극복하기 위한 방법론(Methodology for Overcoming the Problem of Position Embedding Length Limitation in Pre-training Models)' 연구를 위해 작성되었다.

## 1. 사용 방법법

### 1.1. 데이터셋

본 실험을 위해 사용된 데이터셋은 다음과 같다.

* (1) KorQuad v1.0
* (2) AIHub 기계독해
* (3) AIHub 뉴스 기사 기계독해 데이터
* (4) AIHub 일반상식
* (5) AIHub 행정 문서 대상 기계독해 데이터

위의 데이터셋을 아래와 같은 디렉토리 구조를 가지도록 배치한다.

```

.
└── original_data
    ├── aihub_administration
    │   ├── TL_multiple_choice.json
    │   ├── TL_span_extraction_how.json
    │   ├── TL_span_extraction.json
    │   ├── TL_tableqa.json
    │   ├── TL_text_entailment.json
    │   ├── TL_unanswerable.json
    │   ├── VL_multiple_choice.json
    │   ├── VL_span_extraction_how.json
    │   ├── VL_span_extraction.json
    │   ├── VL_tableqa.json
    │   ├── VL_text_entailment.json
    │   └── VL_unanswerable.json
    ├── aihub_commonsense
    │   └── ko_wiki_v1_squad.json
    ├── aihub_mrc
    │   ├── ko_nia_clue0529_squad_all.json
    │   ├── ko_nia_noanswer_squad_all.json
    │   └── ko_nia_normal_squad_all.json
    ├── aihub_newsmrc
    │   ├── TL_span_extraction.json
    │   ├── TL_span_inference.json
    │   ├── TL_text_entailment.json
    │   ├── TL_unanswerable.json
    │   ├── VL_span_extraction.json
    │   ├── VL_span_inference.json
    │   ├── VL_text_entailment.json
    │   └── VL_unanswerable.json
    └── korquad1.0
        ├── dev.json
        └── train.json

```

위 상태에서 preprocess.ipynb를 각각 실행하면 데이터셋이 아래와 같이 전처리된다.

```
.
└── data
    ├── aihub_administration
    │   ├── context.json
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── aihub_commonsense
    │   ├── context.json
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── aihub_mrc
    │   ├── context.json
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── aihub_newsmrc
    │   ├── context.json
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    └── korquad1.0
        ├── context.json
        ├── eval.json
        ├── test.json
        └── train.json

```

### 1.2. 학습

쉘 환경에서 아래 명령어를 입력하면 기본적으로 세팅된 파라미터로 학습이 수행된다.

```shell

python train.py 

```

'train.py'에서 제공하는 파라미터는 다음과 같다.

```
-td, --train_dir: 학습 데이터 경로, default="data/korquad1.0/train.json"
-vd, --eval_dir: 검증 데이터 경로, default="data/korquad1.0/eval.json"
-cd, --context_dir: 컨텍스트 데이터 경로, default="data/korquad1.0/context.json"
-sd, --save_dir: 저장 경로, default="result/korquad1.0_1024/"
-pt, --pretrained_tokenizer: 사전학습된 토크나이저 경로(huggingface 포함), default="klue/bert-base"
-pm, --pretrained_model: 사전학습된 모델 경로(huggingface 포함), default="klue/bert-base"
-e, --num_epochs: 학습 에폭 수, default=100
-p, --patience: early stopping 학습 시 기다릴 에폭 수(early stopping 미설정을 원할 시 -1로 설정), default=3
-b, --batch_size: 배치 사이즈, default=64
-ml, --max_length: 최대 길이, default=1024

```

아래 예시는 논문 비교 실험을 위해 사용한 명령어 일부이다.

```

python train.py \
-td data/aihub_commonsense/train.json \
-vd data/aihub_commonsense/eval.json \
-cd data/aihub_commonsense/context.json \
-sd result/aihub_commonsense_512/ \
-pt klue/bert-base \
-pm klue/bert-base \
-e 100 \
-p 3 \
-b 64 \
-ml 512

```

### 1.2. 시험

주피터 노트북을 통해 'predict.ipynb'를 순서대로 실행하면 예측이 수행된다.

[predict.ipynb](https://github.com/skaeads12/OvercomingPositionEmbeddingLimitation/blob/main/predict.ipynb)

두 번째 쉘에서 'test_dir'와 'context_dir', 'pretrained_models_dir'를 수정하면 각각 시험 데이터와 사전학습 모델의 경로를 수정할 수 있다.

```python

test_dir = "data/aihub_administration/test.json"
context_dir = "data/aihub_administration/context.json"

pretrained_models_dir = "result/roberta/aihub_administration_1024/"

batch_size = 64
max_length = 1024

```

쉘을 순서대로 진행하면, 다섯 번째 쉘에서 성능이 측정된다.

```python

for i, pred in enumerate(preds):

    predictions = pred.predictions
    label_ids = pred.label_ids
    
    exact_match = 0
    f1 = 0
    total = 0

    result = []

    for sample_idx, sample in enumerate(test_set.samples):

        pred_start = predictions[0][sample_idx].argmax(-1)
        pred_end = predictions[1][sample_idx].argmax(-1)

        pred_text = tokenizer.decode(test_set[sample_idx]["input_ids"][pred_start:pred_end], skip_special_tokens=True)

        label_text = context[sample["context"]][sample["answer_start"]:sample["answer_end"]]
        
        exact_match += exact_match_score(pred_text, label_text)
        f1 += f1_score(pred_text, label_text)

        total += 1

    exact_match = round(100.0 * exact_match / total, 4)
    f1 = round(100.0 * f1 / total, 4)

    print("[{}] em score: {}\tf1_score: {}".format(pretrained_models[i], exact_match, f1))


```

```markdown

[checkpoint-147611] em score: 72.8826	f1_score: 90.1524
[checkpoint-163149] em score: 72.732	f1_score: 90.1227
[checkpoint-170918] em score: 72.6915	f1_score: 90.2541

```

# 2. 모델 구조

본 연구를 위해 토큰 분류(Token Classification) 기법을 활용한 질의응답(Question Answering, QA) 모델을 사용하였으며, 코드는 huggingface의 'BertForQuestionAnswering'을 활용하였다.

모델 구조는 아래와 같다.

<img src="https://github.com/skaeads12/OvercomingPositionEmbeddingLimitation/assets/45366231/d0bfd8d6-452f-4d1a-87c6-f54feeec681d" width=65% height=65%>
<img src="https://github.com/skaeads12/OvercomingPositionEmbeddingLimitation/assets/45366231/704edb35-f853-422d-bf30-d64df8591d13" width=50% height=50%>

[이미지 출처: https://www.kaggle.com/code/arunmohan003/question-answering-using-bert]

최대 길이 변환을 위해 사용한 기법은 논문 참조. 소스 코드는 model.py의 set_position_embeddings 메소드 확인.


