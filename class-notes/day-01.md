# Day 01 · 수업 주제 입력

- 날짜: 2026-11-02
- 교안: 아직 미기록 — 파일명 / 페이지 입력
- 수업 실습 노트북: 아직 미기록 — 경로 / 셀 제목 입력
- 직접 작성한 노트북: 아직 미기록

## 수업 중 메모

여기부터 편하게 적으세요. 문장이 완성되지 않아도 괜찮습니다.

- LLM의 구조를 이해하고 코드 실습을 통해 업무에 잘 적용할 수 있도록 확장한다.
- transformer?? 는 무엇인가.
- 1일차는 LLM 모델의 구조를 이해
- AlexNet
- ResNet
- Word2Vec
- Alphago
- Transformers
- GAN/Diffusion
-

### 보충 설명: NLP와 Transformer

- **NLP = Natural Language Processing (자연어 처리)**: 컴퓨터가 사람의 언어를 분석하거나 생성하도록 하는 분야. 번역, 문장 요약, 감정 분석 등이 예다.
- **Transformer (트랜스포머)**: 약자가 아니라 신경망 구조의 이름. Attention(어텐션)을 핵심으로 사용해 입력 요소 사이의 관계를 반영한다. 문장에서는 단어·토큰 사이의 관계를 다루는 데 쓰인다.
- **둘의 관계**: NLP는 언어를 다루는 분야이고, Transformer는 그 분야의 문제를 푸는 데 사용하는 모델 구조 중 하나다.

출처: [Google ML 용어집](https://developers.google.com/machine-learning/glossary#natural-language-processing), [Transformer 원 논문: Attention Is All You Need](https://arxiv.org/abs/1706.03762). 교안 인용이 아닌 보충 설명이다.

### 보충 설명: RNN

- **RNN = Recurrent Neural Network (순환 신경망)**: 문장이나 시계열처럼 순서가 있는 데이터를 한 단계씩 처리하는 신경망이다.
- **핵심 원리**: 현재 입력과 이전 단계의 은닉 상태(hidden state)를 함께 사용해 다음 은닉 상태를 만든다. 은닉 상태는 앞에서 처리한 정보가 담긴 요약이라고 생각하면 된다.
- **예시**: “나는 → 오늘 → 학교에 → 간다”를 차례로 읽으면서 앞부분의 정보를 갱신한다.
- **Transformer와 비교**: 기본 RNN은 이전 상태를 순차적으로 전달하고, Transformer는 Attention으로 토큰 사이의 관계를 계산한다.

출처: [PyTorch 공식 RNN 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html). 교안 인용이 아닌 보충 설명이다.

### 보충 설명: BPE 토크나이저

- **BPE = Byte Pair Encoding (바이트 페어 인코딩)**: 자주 붙어 나오는 조각 두 개를 반복해서 합쳐 **토큰(텍스트 처리 단위)**을 만드는 방식.
- **쉬운 예시**: `l | o | w` → `lo | w` → `low`. 학습 데이터에서 해당 쌍이 자주 나온다고 가정한 예시다.
- **왜 사용할까?** 자주 나오는 표현은 큰 토큰으로, 낯선 단어는 작은 조각으로 표현할 수 있다.
- **기억하기**: 학습할 때 병합 규칙을 만들고, 사용할 때는 그 규칙으로 문장을 토큰화해 숫자 ID로 바꾼다. 토큰 하나가 반드시 단어 하나인 것은 아니다.

출처: [Hugging Face BPE 설명](https://huggingface.co/learn/llm-course/chapter6/5). 이해를 위한 보충 설명이다.

### 보충 설명: Attention과 Self-Attention

- **Attention (어텐션, 주의)**: 지금 처리하는 단어가 다른 단어의 정보를 얼마나 참고할지 계산하는 방식. 관련성에 따라 가중치를 주어 정보를 섞는다.
- **예시**: “철수는 사과를 먹었다. 그는 배가 불렀다.”에서 ‘그는’을 해석할 때 ‘철수’의 정보를 많이 참고한다고 생각하면 된다. 원리를 이해하기 위한 예시이며 실제 가중치는 모델에 따라 달라진다.
- **Self-Attention (셀프 어텐션)**: 같은 입력 안의 단어·토큰들이 서로의 정보를 참고하는 것. 두 용어 모두 약자가 아니다.

출처: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 이해를 위한 보충 설명이다.

#### Q·K·V와 scores·weights·output 구분

- **Q(Query, 질의)**는 찾을 기준, **K(Key, 키)**는 비교할 단서, **V(Value, 값)**는 실제 가져올 정보다.
- 교안의 순서: `scores = Q @ K.T` → `weights = softmax(scores / sqrt(d_k))` → `output = weights @ V`. V까지 곱한 결과는 점수가 아니라 정보를 섞은 출력(context vector)이다.
- **예시**: Q=[1,0], K=[[2,0],[0,2]]이면 scores=[2,0], 스케일링·softmax 후 weights≈[0.804,0.196]. V=[[10,0],[0,10]]을 이 비율로 섞으면 output≈[8.044,1.956]이다.
- [Q·K·V HTML 수업과 직접 바꾸는 예제](../lessons/0009-attention-qkv.html). 교안 Chapter 3 CausalAttention과 대조한 보충 설명이며, 이해·독립 구현 여부는 아직 확인 전이다.

### 보충 설명: RoPE

- **RoPE = Rotary Position Embedding (회전 위치 임베딩)**: Attention 계산에 토큰의 위치·순서 정보를 반영하는 방법.
- **핵심 원리**: 위치에 따라 Query·Key 벡터를 회전시켜, 두 토큰 사이의 상대적인 거리도 Attention 점수에 반영되도록 한다.
- **예시**: 단순하게 각 위치마다 10도씩 회전한다고 생각하면, 1번·3번 토큰은 10도·30도로 회전해 20도 차이가 난다. 4번·6번도 40도·60도로 같은 차이다. ‘두 칸 떨어졌다’는 관계를 표현하는 비유이며, 실제로는 벡터의 성분 쌍마다 서로 다른 회전 속도를 사용한다.

출처: [RoFormer 원 논문](https://arxiv.org/abs/2104.09864). 이해를 위한 보충 설명이다.

### 보충 설명: Self-supervised 학습과 한 칸 shift

- **Self-supervised learning (자기지도학습)**: 사람이 정답을 따로 붙이는 대신, 원본 데이터에서 학습용 정답(target)을 만드는 방식. 다음 토큰 예측은 그중 하나다.
- **예시**: 원본이 `a b c d e`라면 입력은 `a b c d`, target은 시작 위치를 한 칸 뒤로 옮긴 `b c d e`다. `e`는 만들어 낸 답이 아니라 원본의 다음 토큰이다.

| 모델이 참고하는 내용 | 맞힐 다음 토큰 |
| --- | --- |
| a | b |
| a b | c |
| a b c | d |
| a b c d | e |

- **핵심**: 각 위치에서 ‘지금까지의 내용으로 다음 토큰 맞히기’를 배운다. Causal mask(미래를 가리는 마스크)로 뒤쪽 정답을 미리 볼 수 없게 한다.
- 교안 코드: `input_chunk = token_ids[i:i + max_length]`, `target_chunk = token_ids[i + 1:i + max_length + 1]`. 실제로는 문자 대신 토큰 ID를 사용한다.

출처: 교안 `llm_hands_on/llm_from_scratch/Chapter_2_Working_with_Text_Data_KR.ipynb`의 Dataset 구현, [Hugging Face 다음 토큰 예측 설명](https://huggingface.co/docs/transformers/tasks/language_modeling).

### 실습 질문: tiktoken의 encode와 decode

- **빈칸 정답**: 첫 번째는 `encode`, 두 번째는 `decode`.
- **뜻**: `encode`는 문자열 → 토큰 ID 리스트, `decode`는 토큰 ID 리스트 → 문자열로 변환한다.
- **예시**:

```python
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings = tokenizer.decode(integers)
assert strings == text
```

- `allowed_special`은 `<|endoftext|>`를 특수 토큰으로 인식하도록 허용한다. `get_encoding("gpt2")`의 인자는 정확히는 **인코딩 이름**이며, 모델 이름으로 찾는 함수는 `encoding_for_model`이다.
- 버전 확인에 쓰는 하위 모듈은 `import importlib.metadata`로 명시적으로 불러온다. 위 코드는 설명용이며 여기서 실행한 결과는 아니다.

출처: [tiktoken 공식 저장소](https://github.com/openai/tiktoken).

### 실습 질문: Chapter 2 Dataset 전체 빈칸

- **오류 이유**: 실행 코드의 `????`는 Python 문법이 아닌 문제 표시여서 `SyntaxError`가 난다. 주석 속 물음표는 실행에 영향이 없다.
- **예시**: `input_chunk = token_ids[i:i + max_length]`, `target_chunk = token_ids[i + 1:i + max_length + 1]`로 입력·정답을 한 칸 어긋나게 만든다.
- 실행 코드의 **9개 문장, 16개 빈칸 정답과 이유**는 [Dataset 실습 해설](day-01-dataset-answers.md)에 정리했다.

### 추가 실습: DataLoader부터 위치 임베딩까지

- [전체 HTML 수업 목차](../lessons/index.html): Tokenizer 비교 직전까지 교안 코드 셀 7개를 순서대로 연결했다.
- [DataLoader 만들기](../lessons/0005-create-dataloader.html) · [배치 읽기](../lessons/0006-read-batches.html). 예: 길이 4의 샘플을 8개 묶으면 입력 shape는 `(8,4)`다.
- [토큰 임베딩](../lessons/0007-token-embedding.html) · [위치 임베딩](../lessons/0008-position-embedding.html). 예: `(8,4)` ID가 `(8,4,256)` 벡터가 되고, `(4,256)` 위치 벡터를 더해 순서를 반영한다.
- 각 추가 수업에는 빈칸·개념·독립 코딩 문제 7개와 해설이 있다. 수업 제공은 숙달 판정과 구분한다.

### Chapter 3 Attention 셀별 학습

- [셀별 HTML 수업 목차](../lessons/chapter-3.html): 코드 4개, 그림 1개, 시뮬레이션 링크 1개, 빈 Markdown 1개를 각각 안내한다. 실행 코드 빈칸은 총 21곳이다.
- **예시**: Wrapper는 헤드당 2차원·헤드 2개면 전체 4차원. 통합 MultiHeadAttention은 전체 d_out=2·헤드 2개면 헤드당 1차원·출력 2차원이다.
- 원본 셀 4의 저장 출력 `(2,6,4)`는 현재 `d_out=2` 코드와 다르므로 재실행 결과와 구분한다. 핵심 복습은 미래 마스크와 두 구현의 d_out 의미다.

### Chapter 4 GPT 셀별 학습

- [전체 7개 셀 HTML 수업](../lessons/chapter-4.html): 데이터 복습, Attention, LayerNorm·GELU·FeedForward, GPT 조립, 생성 루프, main 실행, 빈 셀 복습을 안내한다. 실행 코드 빈칸은 총 19곳이다.
- **예시**: 입력 ID `(2,3)` → 임베딩 `(2,3,8)` → 블록 → 어휘 크기 32의 logits `(2,3,32)`. 마지막 위치 점수에서 다음 ID를 고른다.
- **기억하기**: 시작 토큰 4개에 새 토큰 10개를 붙이면 14개다. `eval()`은 dropout 평가 모드, `no_grad()`는 기울기 기록 제어다. 이 교안 모델은 무작위 초기화이며 아직 사전학습 가중치를 로드하지 않는다.

## 오늘 핵심 3줄

세 번째 실습 복습: [GPTDatasetV1 HTML 수업과 7개 연습 문제](../lessons/0004-gpt-dataset.html). 예: 9토큰에서 길이=4, stride=2이면 시작점 0·2·4의 샘플 3개를 만든다. __len__은 샘플 수, __getitem__은 같은 번호의 입력·정답 쌍을 반환한다.

두 번째 실습 복습: [파일 읽기·토큰화·한 칸 shift HTML 수업과 7개 연습 문제](../lessons/0003-text-to-next-token.html). 예: `[10,20,30,40,50]`에서 x는 앞의 4개, y는 한 칸 뒤의 4개다. 슬라이스 끝 인덱스는 포함하지 않는다.

실습 복습: [BPE·encode/decode HTML 수업과 7개 연습 문제](../lessons/0002-bpe-encode-decode.html). 예: `"Hello,"` → `[15496, 11]` → `"Hello,"`. ID는 토큰의 식별 번호다.

수업 후 내 말로 채우세요.

1.
2.
3.

## 코드 흐름과 이유

| 단계 또는 함수 | 내 말로 설명 | 입력 → 출력 shape (해당 시) |
| --- | --- | --- |
| | | |

## 강사의 시험 관련 언급

- 언급 내용과 교안 위치:
- 내 예상 또는 아직 확인할 점:

## 질문과 오류

- 이해되지 않는 점:
- 오류 메시지와 발생한 코드 위치:
- 원인 / 수정 방법 / 수정 후 결과:

## 자료 없이 재작성

- 오늘 다시 작성할 작은 기능:
- 작성한 노트북과 셀:
- 결과: 미시도
- 참고한 자료나 받은 힌트:
- 이 코드가 필요한 이유를 내 말로 설명:
- 다음번에 보완할 점:

교안 확인 전의 선택 진단: [텐서 shape 첫 과제](../lessons/0001-tensor-shapes.html). 실제 오늘 수업에서 다룬 내용으로 바꿔도 됩니다.

## 복습 예약

| 시점 | 복습할 질문 또는 구현 | 실제 수행일 | 결과와 받은 도움 |
| --- | --- | --- | --- |
| 다음 수업일 (Day 2) | | | |
| 3수업일 후 (Day 4) | | | |
| 7수업일 후 (Day 8) | | | |

## 마무리

- [ ] 핵심을 내 말로 설명했다.
- [ ] 자료를 닫고 재작성을 시도했다.
- [ ] 오류와 도움받은 부분을 기록했다.
- [ ] 다음에 복습할 주제를 골랐다.
