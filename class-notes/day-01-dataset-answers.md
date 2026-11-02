# Day 01 · Chapter 2 Dataset 빈칸 정답과 이유

대상: [Chapter_2_Exercise_Dataset.ipynb](../llm_hands_on/Chapter_2_Exercise_Dataset.ipynb). 셀 번호는 실행 횟수 `[번호]`가 아니라 **파일 맨 위부터 코드·Markdown 셀을 모두 센 순서**다.

## 왜 실행이 안 될까?

`????`와 `?????`는 직접 채우는 문제 표시다. Python 문법이 아니어서 코드에 남아 있으면 `SyntaxError: invalid syntax`가 발생한다. Python은 셀 전체의 문법을 먼저 확인하므로, 같은 셀 앞부분에 정상적인 코드가 있어도 실행되지 않는다.

예: `tokenizer.????(text)`는 잘못된 문법이고, `tokenizer.encode(text)`처럼 실제 메서드 이름을 넣어야 한다. `# 힌트: ????`처럼 주석에 있는 물음표는 실행에 영향을 주지 않는다.

## 정답 한눈에 보기

실행 코드 기준 **9개 문장, 총 16개 빈칸**이다. 물음표가 5개인 곳도 빈칸 한 곳으로 센다.

| 순서 | 셀 | 완성 코드 | 빈칸 답 순서 |
| --- | --- | --- | --- |
| 1 | 1 | `integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})` | `encode` |
| 2 | 1 | `strings = tokenizer.decode(integers)` | `decode` |
| 3 | 2 | `y = enc_sample[1:context_size + 1]` | `1`, `1` |
| 4 | 3 | `input_chunk = token_ids[i:i + max_length]` | `i`, `i` |
| 5 | 3 | `target_chunk = token_ids[i + 1:i + max_length + 1]` | `i`, `1`, `i`, `1` |
| 6 | 3 | `return self.input_ids[idx], self.target_ids[idx]` | `idx`, `idx` |
| 7 | 4 | `DataLoader`의 첫 번째 인자에 `dataset` | `dataset` |
| 8 | 6 | `embedding_layer = torch.nn.Embedding(vocab_size, output_dim)` | `vocab_size`, `output_dim` |
| 9 | 6 | `token_embeddings = embedding_layer(input_ids)` | `input_ids` |

## 1. 문자열을 숫자로: encode

```python
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
```

**이유:** `encode`는 문자열을 토큰 ID 리스트로 바꾼다. 모델에는 글자 자체가 아니라 숫자 ID를 전달한다. `allowed_special`은 해당 문자열을 특수 토큰으로 인식하도록 허용한다.

**예시:** `tokenizer.encode("Hello")`는 GPT-2 인코딩에서 `[15496]`이다. 문장이나 공백이 달라지면 분리 결과도 달라질 수 있다.

## 2. 숫자를 문자열로: decode

```python
strings = tokenizer.decode(integers)
```

**이유:** `decode`는 토큰 ID 리스트를 문자열로 복원한다. `encode`와 반대 방향이다.

**예시:** 같은 GPT-2 인코딩에서 `tokenizer.decode([15496])`는 `"Hello"`다. 전체 원문 복원은 `assert strings == text`로 확인한다.

## 3. 다음 토큰을 정답으로: 1, 1

```python
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
```

**이유:** `x`의 각 위치에서 다음 토큰을 예측하므로 `y`는 원본에서 한 칸 뒤부터 가져온다. 시작뿐 아니라 끝도 1 늘려 입력과 정답의 길이를 같게 한다. 슬라이스 `[시작:끝]`에서 **끝 위치는 포함하지 않는다.**

**예시:** `enc_sample = [10, 20, 30, 40, 50]`, `context_size = 4`이면 `x = [10, 20, 30, 40]`, `y = [20, 30, 40, 50]`이다. `50`도 원본 데이터에 있는 값이다.

## 4. 입력 조각 만들기: i, i

```python
input_chunk = token_ids[i:i + max_length]
```

**이유:** 현재 시작점 `i`부터 `max_length`개를 잘라 입력 샘플 하나를 만든다. 시작점은 반복문이 `stride`만큼 옮긴다.

**예시:** `token_ids = [10, 20, 30, 40, 50, 60]`, `i = 1`, `max_length = 3`이면 `[1:4]`이므로 `[20, 30, 40]`이다.

## 5. 정답 조각 만들기: i, 1, i, 1

```python
target_chunk = token_ids[i + 1:i + max_length + 1]
```

**이유:** 입력 조각의 시작·끝보다 각각 한 칸 뒤를 선택한다. 길이는 입력과 같고, 내용은 각 위치의 다음 토큰이다. 그래서 길이 `max_length`의 입력·정답 쌍을 만들려면 원본 토큰이 최소 `max_length + 1`개 필요하다.

**예시:** 위 데이터에서 `i = 1`, `max_length = 3`이면 입력은 `[20, 30, 40]`, 정답은 `[30, 40, 50]`이다. 모델이 미래 정답을 보지 못하게 하는 Causal mask는 이후 모델 쪽에서 다루며, 이 Dataset이 만드는 것은 입력·정답 쌍이다.

## 6. 같은 샘플의 입력·정답 반환: idx, idx

```python
return self.input_ids[idx], self.target_ids[idx]
```

**이유:** `__getitem__`은 요청받은 번호 `idx`의 샘플을 꺼낸다. 입력과 정답 모두 같은 번호를 사용해야 한 쌍이 맞는다. **다음 토큰으로 미는 작업은 이미 target_chunk에서 했다.** 여기서 `idx + 1`을 사용하면 다른 샘플을 꺼내게 된다.

**예시:** `dataset[0]`을 요청하면 0번 입력과 0번 정답을 함께 반환한다.

## 7. 배치로 꺼낼 대상 전달: dataset

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
)
```

**이유:** Dataset은 샘플을 만들고, DataLoader는 그 샘플들을 배치로 묶어 꺼낸다. 첫 번째 인자로 방금 만든 `dataset` 객체를 전달한다.

**예시:** 샘플 8개를 `batch_size=4`로 묶으면 배치 2개다. 이 교안에서 샘플 길이가 4이면 각 입력 배치의 shape는 `(4, 4)`다.

## 8. 임베딩 표 만들기: vocab_size, output_dim

```python
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

**이유:** 임베딩은 토큰 ID를 실수 벡터로 바꾸는 학습 가능한 표다. 첫 인자는 표의 행 수(토큰 종류 수), 둘째 인자는 한 행의 길이(벡터 차원)다. ID는 0부터 시작한다.

**예시:** `Embedding(6, 3)`은 shape `(6, 3)`인 표를 만들며, ID `0~5` 각각을 숫자 3개짜리 벡터에 대응시킨다. ID `6`은 범위를 벗어난다.

## 9. 토큰 ID를 벡터로 바꾸기: input_ids

```python
token_embeddings = embedding_layer(input_ids)
```

**이유:** 만든 임베딩 표에 정수 토큰 ID를 넣으면 각 ID에 해당하는 행을 꺼낸다. 임베딩은 이 단계에서 단어 뜻을 완성해 주는 것이 아니라, 학습할 벡터를 제공한다.

**예시:** `input_ids = torch.tensor([2, 3, 5, 1])`, `Embedding(6, 3)`이면 4개 ID가 각각 3차원 벡터가 되어 출력 shape는 `(4, 3)`이다.

## 빈칸 없는 다음 셀도 이해하기

7번째 셀은 이미 완성되어 있다. 배치 8개, 토큰 길이 4, 임베딩 차원 256이라는 예시로 보면:

```text
inputs                              (8, 4)
token_embedding_layer(inputs)       (8, 4, 256)
torch.arange(4)                     [0, 1, 2, 3]
pos_embedding_layer(torch.arange(4)) (4, 256)
token_embeddings + pos_embeddings   (8, 4, 256)
```

토큰 임베딩에 위치 임베딩을 더해 위치 정보를 반영한다. 위치 임베딩은 배치마다 공통으로 적용된다(브로드캐스팅). 이 셀은 **학습 가능한 위치 임베딩을 더하는 방식**이며, 앞서 질문한 RoPE의 회전 방식과 구분한다.

## 빈칸 외에 확인할 점

- **import:** 버전 출력에 필요한 모듈을 확실히 가져오도록 첫 줄은 `import importlib.metadata`로 적는다.
- **붙는 문자열:** 원본의 `"...terraces" "of..."`는 자동 연결되어 `terracesof`가 된다. 띄어쓰기를 의도했다면 `"...terraces "`처럼 공백을 넣는다. 이것은 문법 오류는 아니다.
- **실행 순서:** 첫 셀부터 실행한다. `tokenizer`, `raw_text`, `GPTDatasetV1`, `create_dataloader_v1`을 만든 뒤에 사용하는 셀을 실행한다.
- **환경·인터넷:** `ModuleNotFoundError`는 패키지 누락, `NameError`는 앞선 정의 미실행 등을 확인한다. 텍스트 다운로드와 마지막 토크나이저 비교는 인터넷·모델 접근 권한 등에 영향을 받을 수 있다. `????` 문법 오류와 별개다.
- **추가 빈칸:** 5·7·10번째 코드 셀 및 뒤의 Markdown에는 채울 실행 코드 빈칸이 없다. 주석의 물음표는 남아 있어도 된다.

## 확인 범위와 복습

정답을 메모리상 코드에 대입해 코드 셀 8개의 Python 문법을 검사하고, 숫자 리스트로 입력·정답 슬라이싱 결과를 확인했다. 원본 노트북은 수정하지 않았다. PyTorch·tiktoken 실행, 데이터 다운로드, 모델 토크나이저 로딩을 포함한 전체 실행은 하지 않았다.

직접 빈칸을 채운 뒤 커널을 재시작하고 위에서 아래로 실행한다. 이 문서를 닫고 `input_chunk`, `target_chunk`, `__getitem__` 세 줄을 다시 써 보고, 특히 **target은 왜 +1이고 idx는 왜 그대로인지** 설명해 본다. 막히는 부분은 에이전트에게 질문한다.

## 참고 자료

- 위 정답은 대상 노트북의 코드·힌트와 대조했다.
- [tiktoken 공식 저장소](https://github.com/openai/tiktoken): encode/decode 사용 예시.
- [PyTorch Dataset·DataLoader](https://docs.pytorch.org/docs/stable/data.html): 샘플 조회와 배치 구성.
- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html): 토큰 ID를 벡터로 바꾸는 표.
