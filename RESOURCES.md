# PyTorch 실기 학습 자료

## Knowledge

- Day 2 Attention 수업 근거: `llm_hands_on/pdf/app_sp_slides_20261013.pdf` **104~120쪽** 본문을 2026-11-03 확인했다. 문맥·QKV 역할·학습 의미·점수·스케일·가중합·softmax·mask·multi-head를 Chapter 3 셀 1~5와 대조했다. [시각화 수업](lessons/0026-day02-attention-journey.html), [복습 카드](reference/day02-attention-journey.html). PDF 109쪽의 자기 자신 우선 주목은 항상 성립하는 성질로 일반화하지 않는다. 원 논문 §3.2.1~2 및 PyTorch Linear로 보충 검증했다.

- [AI인증시험_출제포인트.pdf](AI인증시험_출제포인트.pdf) 1쪽: 2026-11-03 본문을 확인했다. LLM·RAG·Data·Vision·On-device의 평가 관점과 2026-11-17 시험 표기를 담고 있다. 출제 전 `EXAM-SCOPE.md`에서 사용자 확정 제외 범위와 함께 읽는다.

- [수업 자료 안내](llm_hands_on/README.md): 사용자가 받은 강의 자료 저장소. 수업 정리와 과제 출제 시 최우선으로 읽고 당일 실제 진도에 해당하는 노트북을 확인한다.
- `llm_hands_on/Chapter_*.ipynb`: 벡터 공간, Dataset/DataLoader, Attention, GPT, 사전학습, 분류 파인튜닝, LoRA, instruction tuning 및 DPO 실습. 파일 존재와 README 목차를 확인했으며 실제 시험 포함 여부는 강사 공지로 확인한다.
- [현재 존재하는 슬라이드](llm_hands_on/pdf/app_sp_slides_20261013.pdf): PDF 본문은 아직 검토하지 않았다. 자료 README의 PDF 링크는 다른 파일명을 가리키므로 실제 파일을 기준으로 참조한다.
- `llm_hands_on/llm_from_scratch/`: 챕터별 참고 노트북과 추가 자료. 당일 실습을 이해하는 데 필요한 부분을 찾아 읽는다.
- [PyTorch 공식 Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
  데이터부터 학습·평가까지 연결하는 흐름을 확인할 때 사용한다. 예제 데이터는 FashionMNIST이며 실제 시험 데이터셋으로 단정하지 않는다.
- [PyTorch 공식 Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
  텐서 생성, 속성, 연산을 설명하고 첫 과제의 shape 조작을 확인할 때 사용한다.

공식 문서 확인일: 2026-11-02. 실습 환경의 설치 버전이 확인되면 해당 버전과 교안을 대조한다.

- [tiktoken 공식 저장소](https://github.com/openai/tiktoken) 및 [API 구현](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py): encode/decode와 특수 토큰 옵션을 설명할 때 사용한다. 첫 HTML 실습의 API 근거다.
- [Hugging Face BPE 과정](https://huggingface.co/learn/llm-course/chapter6/5): 병합 규칙 학습과 토큰화 적용을 구분하고 byte-level BPE를 설명할 때 사용한다.

- [Python 3.12 리스트·슬라이싱](https://docs.python.org/3.12/tutorial/introduction.html#lists) 및 [파일 읽기·쓰기](https://docs.python.org/3.12/tutorial/inputoutput.html#reading-and-writing-files): 두 번째 실습의 인덱스 범위와 파일 모드를 설명할 때 사용한다.
- [Requests Quickstart](https://requests.readthedocs.io/en/latest/user/quickstart/): 다운로드 응답, HTTP 오류 확인, timeout의 의미를 확인할 때 사용한다.

- [PyTorch DataLoader·Dataset](https://docs.pytorch.org/docs/stable/data.html): 배치 옵션과 Dataset 조회 동작을 확인할 때 사용한다.
- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) 및 [broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html): 토큰 ID 조회, 출력 shape, 위치 임베딩 합산을 설명할 때 사용한다.
- [Python iter·next](https://docs.python.org/3.12/library/functions.html#iter): 이터레이터 생성, 조회 진행 및 StopIteration을 설명할 때 사용한다.

- [Attention Is All You Need §3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1): scaled dot-product attention 수식과 Q·K 비교, V 가중합의 역할을 확인할 때 사용한다. 실습의 변수명은 `llm_hands_on/Chapter_3_Excercise_Attention.ipynb`의 CausalAttention과 대조한다.

- [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) 및 [ModuleList](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html): Chapter 3의 Q·K·V 투영과 헤드 모듈 등록을 설명할 때 사용한다.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) 및 [LLM Visualization](https://bbycroft.net/llm): 교안 셀 6의 시각화 관찰 과제에 사용한다. 코드 실행 결과와 외부 화면에서 관찰한 결과는 구분한다.

- [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), [GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html), [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html): Chapter 4의 정규화, tanh 근사 활성화, 생성 시 기울기 기록 제어를 확인할 때 사용한다. eval 동작은 [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)과 대조한다.

## Wisdom (Communities)


- 현재 수강 중인 AI Specialist 수업의 강사와 동료: 시험 범위·허용 도구·교안의 의도를 확인하는 창구다. 확인할 질문을 정리하고 사용자가 직접 문의한다.

## Gaps

- 강의 자료 경로는 확인했다. 당일 진도, 실제 평가 범위·공지, 설치된 라이브러리 버전은 추가 확인이 필요하다.
- Python 선수 지식과 하루 학습 가능 시간은 첫 진단 및 대화로 확인한다.
