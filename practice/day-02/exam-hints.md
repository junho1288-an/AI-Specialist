# Day 2 · 시험 출제 힌트 모음

확인일: 2026-11-03. 실제 기출·유출 문제가 아니라 담당자 안내, 출제 포인트 PDF, 현재 교안을 모아 정리한 대비 기준이다.

## 지금까지 확인된 안내

| 안내 | 근거 | 준비 방향 |
| --- | --- | --- |
| 전체 코드를 통째로 암기하기보다, 주어진 파라미터로 함수 내부를 구현 | 사용자가 전달한 시험 담당자 안내, 2026-11-03 | 함수의 입력·처리·반환 관계를 이해하고 구현 |
| Jupyter Notebook 코드 작성 실기, 교안에서 출제 예정 | 저장소 README의 인증 평가 안내 | 교안의 함수·레이어·데이터 흐름 중심 연습 |
| 레이어의 의미·구성·활용, Tensor 흐름, 학습 대상과 데이터 구성을 점검 | 출제 포인트 PDF 1쪽 LLM 항목 | API 이름뿐 아니라 shape와 처리 이유도 설명 |
| Chapter 2의 Tokenizer 비교부터 이후 내용 제외 | 사용자 확인 | 그 앞의 토큰화·Dataset·배치·임베딩에 집중 |
| 온라인 파일·모델 다운로드 절차 제외 | 사용자 확인 | 제공된 문자열·텐서로 구현 시작 |

제공되는 코드 양, 함수명·파라미터명 제공 범위, 문항 수·배점·제한 시간·도움말 허용 여부는 아직 확인되지 않았다. 함수 내부 구현 형식이어도 필요한 라이브러리 사용법은 알아야 한다.

## 안내를 코드 문제로 해석하면

아래는 **연습용 예시**이며 실제 시험 문항은 아니다.

```python
def make_next_token_pair(token_ids, start, max_length):
    # start 위치부터 max_length개 입력과
    # 그보다 한 칸 뒤의 정답을 만들어 두 텐서로 반환
    # 구현할 부분
    pass
```

조건: `token_ids`는 정수 리스트, `start >= 0`, `max_length > 0`, `start + max_length < len(token_ids)`이며 torch는 제공되어 있다고 가정한다.

제공 예시: `token_ids=[10,20,30,40,50,60]`, `start=1`, `max_length=3`.

반환해야 할 내용: 입력 `[20,30,40]`, 정답 `[30,40,50]`, 각각 shape `(3,)`의 정수 텐서. 특정 숫자나 길이를 고정하지 않고 **받은 인자**로 계산해야 한다. 구현 정답은 먼저 시도한 뒤 확인한다.

여기서 파라미터는 함수의 `start`, `max_length` 같은 **매개변수**를 뜻한다. Attention의 W처럼 학습되는 모델 파라미터와 문맥을 구분한다.

## 교안에서 우선 연습할 함수·메서드

아래는 **교안과 PDF를 바탕으로 한 대비 우선순위**다. 출제 확정이나 확률을 뜻하지 않는다.

| 부분 | 주어질 수 있는 입력·설정 | 내부에서 할 수 있어야 하는 일 | 주의점·근거 |
| --- | --- | --- | --- |
| 다음 토큰 쌍 | token_ids, max_length, stride 등 | 윈도 추출, 한 칸 뒤 정답, 텐서화 | Chapter 2 셀 2~3. 범위·종료 인덱스 |
| Dataset 메서드 | txt, tokenizer, max_length, stride, idx | 샘플 저장, 길이 반환, 같은 idx의 입력·정답 반환 | Chapter 2 셀 3. i와 idx 구분 |
| DataLoader 생성 함수 | txt, batch_size, shuffle, drop_last 등 | 올바른 데이터셋 생성, 받은 옵션 전달 | Chapter 2 셀 4. 전역 text 사용 금지 |
| 임베딩 처리 | input_ids, vocab_size, output_dim | 토큰별 벡터 조회, 필요 시 위치 벡터 합산 | Chapter 2 임베딩 셀. ID 개수와 어휘 수 구분 |
| CausalAttention.__init__ | d_in, d_out, context_length, dropout | QKV 투영·dropout·미래 마스크 구성 | Chapter 3 셀 1. 생성자와 forward 역할 구분 |
| CausalAttention.forward | self에 정의된 레이어, x | QKV → 점수 → mask → scale·softmax → V 가중합 | Chapter 3 셀 1. dim=-1, transpose 축 |
| MultiHeadAttention.forward | x, 헤드 수·차원 등 설정 | 헤드 분리, 헤드별 계산, 결합·출력 투영 | Chapter 3 셀 4. d_out % num_heads 조건 |
| GPT 구성·forward | cfg, in_idx | 임베딩, 블록, 정규화, 어휘별 logits 연결 | Chapter 4 셀 3~4. 단일 헤드 연습 후 진행 |
| 생성 함수 | model, idx, max_new_tokens, context_size | 문맥 자르기, 마지막 위치 점수에서 선택, 입력에 추가 | Chapter 4 셀 5. 이미 완성된 model 활용 |

PDF의 RAG·Data·Vision·On-device도 전체 대비 대상이다. 위 표는 현재 살펴본 Chapter 2~4의 구현 연습 목록이며 전체 시험 범위를 대신하지 않는다. 다른 분야는 해당 교안·진도에 맞춰 같은 방식으로 정리한다.

## 지금 자신의 코드에서 먼저 고칠 습관

1. **받은 인자를 그대로 사용하기.** 제출 코드의 helper는 `txt`를 받았지만 전역 `text`를 읽었다. 시험 함수가 다른 데이터로 호출되면 틀릴 수 있다.
2. **입력값이 바뀌어도 동작하게 만들기.** max_length가 4일 때만 되는 코드보다, 주어진 max_length를 활용한 코드로 연습한다. 단, 문제에서 고정한 값은 조건대로 사용한다.
3. **반환 규격부터 확인하기.** 어떤 값·순서·shape·dtype을 반환해야 하는지 먼저 파악한다. print만 하고 return을 빠뜨리지 않는다.
4. **축과 데이터 관계를 확인하기.** Attention은 (B,T,T) 비율과 (B,T,D) 정보가 결합한다. 다음 토큰 정답은 입력보다 한 칸 뒤다.

이는 대비 방법이며 실제 채점기가 숨은 테스트를 쓰는지, 부분 점수를 어떻게 주는지는 미확인이다.

## 앞으로의 연습 방식

- 문제마다 함수 시그니처·입력 조건·필요 라이브러리·반환 규격·예시를 제공한다.
- 처음에는 함수 한 개를 구현하고, 익숙해지면 관련 함수 여러 개를 연결한다.
- 서로 다른 입력·길이·설정으로 확인하며, 정답 구현은 첫 시도 전 공개하지 않는다.
- 완료 기준은 새 커널 실행, 예시와 다른 입력에서의 동작, 핵심 처리 이유 설명이다. 시간·점수는 연습 기준으로만 표시한다.

근거: [시험 범위 기준](../../EXAM-SCOPE.md), [출제 포인트 PDF 1쪽](../../AI인증시험_출제포인트.pdf), [README](../../README.md), `llm_hands_on/Chapter_2_Exercise_Dataset.ipynb`, `Chapter_3_Excercise_Attention.ipynb`, `Chapter_4_Excercise_GPT.ipynb`. 자신의 제출 피드백은 [Chapter 2 채점 보고서](chapter-02-final-feedback.md) 참고.
