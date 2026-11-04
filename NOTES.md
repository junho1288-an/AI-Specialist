# 지도 메모

## 현재 수업일

- 2026-11-04 출제 포인트 PDF와 실습 코드 기반의 Jupyter 모의시험을 요청받았다. `practice/day-03/llm-mock-exam.ipynb`(기본 14문항 90점·Chapter 7 확장 2문항 10점), `vision-mock-exam.ipynb`(12문항 100점), `mock-exam-guide.md`를 작성했다. 빈칸·부분 구현, 독립 실행 가능한 문제별 확인 셀, 이유·shape 설명란을 제공했다. 제작자 검증과 사용자 숙달은 구분하며 현재 사용자 제출 전이다. 기본 LLM Chapter 3~6 대비를 먼저 하고 Chapter 7은 확장으로 구분한다.

- 시험 형식 최신 확인(2026-11-04): 사용자가 `????` 빈칸 넣기 또는 class·function이 제공된 상태에서 부분 코딩이라고 구체화했다. 시험 대비는 전체 코드 독립 작성보다 문맥·API·shape·축을 이해하고 해당 구간을 완성하는 문제 중심으로 한다. 20시간은 이 형식의 핵심 대비·오답 교정 가안으로 재평가하며 실제 풀이 전 충분함을 단정하지 않는다.

- 사용자는 LLM Chapter 3~6와 Vision 4개 실습을 내일 4시간·토요일 10시간·일요일 6시간에 준비 가능한지 평가를 요청했다. `practice/day-03/llm-vision-20-hour-plan.md`에 20시간 핵심 대비 가안을 작성했다. 전체 독립 구현 완료는 이 시간 내 보장하지 않으며, 공통 학습 루프 재사용·shape·주요 함수 구현 중심으로 진행한다. 설치·전체 모델 학습·휴식은 실제 집중 시간과 구분한다.

- Day 3 사용자는 강의 그림 표기를 Batch×Channel×Width×Height로 맞추길 원한다. 그림 설명은 이 순서를 표시하되 PyTorch의 실제 BCHW·CHW 축 순서와 구분한다. 정사각형 CIFAR에서 수치가 같아도 실제 API 규약을 BCWH로 설명하지 않는다.

- Day 3 사용자는 ResNet18 입력부터 10클래스 분류까지 맨 위부터 차근차근, 단계별 shape 문제와 그림을 원한다. `lessons/0028-day03-resnet18-shape-journey.html`에 9단계 안내를 제공했다. 현재 시작점은 1단계 RGB 이미지이며 답변 전이다. 한 번에 다음 단계 설명을 몰아주지 않고 사용자의 답을 확인하며 진행한다. 노트북 수정 stem 기준으로 B=2 실제 shape를 검증했다.

- Day 3 현재 실습은 `vision/01_ResNet18_CIFAR10.ipynb`. 29개 셀의 구성·함수 목록을 확인했다. 사용자는 직접 실습하며 모르는 부분을 질문할 예정이므로 요청한 셀 중심으로 간결한 설명·예시·shape를 제공하고 Day 3 노트에 기록한다. 전체 학습·다운로드·정답 일괄 제공은 요청하지 않았다.

- 2026-11-04 사용자가 시험관의 “교안 코드도 시험에 나온다”는 안내를 전달했다. Vision PDF의 code snippet도 별도 수집·설명한다. `vision/01_Intro-code-snippets.md`에 Intro 17쪽의 API 네 개와 shape·예제·확인 문제를 정리했다. 원문과 보완 코드를 구분하고, 이후 교안을 볼 때도 코드 페이지를 기록한다.

- 2026-11-03 시험 담당자 안내를 사용자가 전달했다: 통짜 코드 암기보다 파라미터를 제공한 함수 내부 구현 형식. 이후 시험 대비 문제는 시그니처·입력 조건·반환 규격을 제공하고 본문을 작성하게 한다. `EXAM-SCOPE.md`, `practice/day-02/exam-hints.md`에 반영했다. 실제 제공 코드 양·문항 수·채점 방식은 미확인이다.

- 시험 범위는 `EXAM-SCOPE.md`가 기준이다. 사용자는 Chapter 2의 Tokenizer 비교 이후와 온라인 파일 다운로드를 시험에서 제외한다고 확정했다. `AI인증시험_출제포인트.pdf` 1쪽을 읽었으며 PDF에 적힌 시험일은 2026-11-17이다.

- **Day 3 / 2026-11-04 (Asia/Seoul) · Vision**. 사용자가 3일차 시작과 교안 위치 `vision/`을 확정했다.
- 새 질문·설명은 `class-notes/day-03.md`, 실습은 `practice/day-03/`에 기록한다. 이전 주제의 후속 질문도 현재 작성일 노트에서 이전 기록을 연결한다.
- 새 HTML 수업은 기존 순번을 이어가되 파일명에 `day03`를 포함하고 제목·배지는 Day 3로 표시한다. 새 참고 자료도 `reference/day03-topic.html`처럼 일자를 구분한다. 공통 assets는 재사용한다.
- Vision 교안은 `01_Intro.pdf`(31쪽), `02_DNN_CNN.pdf`(43쪽), `03_Attention.pdf`(59쪽), `04_Applications.pdf`(101쪽)다. 각 첫 2쪽과 Intro 2쪽의 일정만 확인했다. 실제 당일 도달 페이지·실습 노트북은 미확인이다. Vision의 세부 일정상 1일차와 전체 과정의 Day 3을 혼동하지 않는다.
- 기존 Day 1~2 산출물은 보존한다. Day 2 Chapter 2 최종 실기는 제출·채점되었으며 아래 기록과 피드백 보고서에서 상태를 확인한다.

## 기존 지도 기록

- Day 2 Attention 이론 연결을 다시 요청했다. 토큰화 이후 Q·K·V의 필요성과 최종 목적을 어려워한다. `lessons/0026-day02-attention-journey.html`에 Chapter 3 실제 입력·시드 기반 인터랙티브 시각화, 역할별 실험, 확인 문제를 제공했다. 교안 PDF 104~120쪽 본문을 확인했다. 역할·shape·미래 마스크는 다음 수업일 및 3·7수업일 후 복습하며, 숙달 판정은 답변·구현 후에 한다.

- Day 2 제출 `practice/day-02/chapter-02-final.ipynb` 채점: 범위 밖 문제 3 제외, 65/90점(72.2/100). 보고서는 `practice/day-02/chapter-02-final-feedback.md`. 원본 셀 8개가 실행되지만 `enc_text[:50]`과 helper의 `txt` 대신 전역 `text` 사용이 의미 오류였다. Dataset 기본 로직과 임베딩 shape는 추가 검사로 확인했다. 문제 8 미제출, 구두 설명·독립 작성 여부는 미확인이다.

- Day 2 Chapter 2 문제지는 `practice/day-02/chapter-02-final-quiz.md`다. 초기 100점·선택 3문제에서 범위 확정 후 수정했다. 현재는 문제 3을 제공 문자열 준비(비채점)로 바꾸고 Tokenizer 비교 이후 선택 문제를 제거했다. 채점 대상 9문제·90점이며 100점 환산한다. 제출과 보고서는 같은 폴더에 있다.

- Day 2 사용자는 vocab_size=6·입력 ID 개수 4·output_dim=3과 Embedding의 역할을 혼동한다고 밝혔다. `lessons/0025-day02-embedding-lookup.html`에서 실제 weight 표와 ID별 조회를 시각화했다. 이해와 독립 구현은 제출 후 확인한다.

- Day 2 사용자는 batch 개념을 어렵다고 밝혀 `lessons/0024-day02-batch.html`에 학생·이미지·토큰 예시, drop_last 조작, 7개 확인 문제를 제공했다. 샘플·배치·토큰 길이 구분의 이해는 제출 후 확인한다.

- 사용자는 12일간 수업 내용을 정리하며 매일 과제와 설명을 받기 원한다.
- 목표는 아무것도 보지 않고 코드를 작성하는 것이다. 코드의 이유를 함께 설명하고 도움의 양을 점차 줄인다.
- 수업 자료는 `llm_hands_on/`에 있다. `RESOURCES.md`의 자료 안내를 따라 당일 해당 노트북부터 확인한다. README와 파일 목록상 NLP/LLM 중심이며, 초기 MNIST 예상보다 실제 교안을 우선한다. 오늘 진도, Python 기초 수준, PyTorch/Jupyter 버전, 하루 복습 가능 시간, 시험일·제한 시간은 미확인이다.
- Day 1 기초 진단은 아직 제출되지 않았다. 현재 숙달로 인정한 개념은 없다.
- Chapter 4의 7개 셀별 수업은 `lessons/chapter-4.html`, 문서는 0017~0023이다. 빈칸 19곳, 모듈 정의와 생성 루프를 설명하고 작은 vocab=32 모델 과제를 제공했다. 원본 main의 큰 모델 실행과 학습된 모델 로딩을 구분한다. 사용자 숙달은 미확인이다.
- Chapter 4 완성 코드 6개는 19개 빈칸만 채운 원본과 AST로 대조했다. PyTorch 2.5.1에서 정의 로딩, 작은 실기 예제 6개, LayerNorm·GELU 대조, logits shape·인과성, greedy 선택·문맥 자르기·no_grad 검사를 통과했다. 원본 큰 설정 main은 실행하지 않았고, 원본 노트북 해시는 유지했다. 브라우저 UI 실행 검증은 별도다.
- Chapter 3의 7개 셀별 수업은 `lessons/chapter-3.html`, 개별 문서는 0010~0016이다. 셀 4는 현재 d_out=2와 저장된 마지막 축 4가 불일치한다. 현재 코드의 출력은 2로 설명하고, Wrapper의 헤드당 d_out과 통합 구현의 전체 d_out을 구분한다. 학습자 숙달은 미확인이다.
- Chapter 3 완성 코드 4개는 교안의 21개 빈칸만 채운 코드와 AST로 대조하고 PyTorch 2.5.1에서 순차 실행했다. 실행 shape는 셀 2 (2,6,2), 셀 3 (2,6,4), 셀 4 (2,6,2). 미래 차단·버퍼 등록·단일 헤드·Wrapper·통합 헤드 실기 확인도 통과했다. 원본 노트북의 해시는 유지되었으며 브라우저 UI 실행 검증은 별도다.
- 사용자는 Q·K 내적과 V 가중합에서 scores·weights·output의 의미를 혼동한다고 밝혔다. `lessons/0009-attention-qkv.html`에 수치 예제, Q/V 조절, shape 구분과 7개 연습을 제공했다. 설명을 제공한 상태이며 오개념이 교정되었다는 증거는 아직 없다.
- `lessons/index.html`에 Tokenizer 비교 전 코드 셀 7개 수업을 연결했다. 추가 수업 0005~0008은 DataLoader 생성, 배치 조회, 토큰 임베딩, 위치 임베딩이다. 각 7개 문제를 제공했으며 학습자 제출·독립 구현 여부는 아직 확인하지 않았다.
- 추가 수업 0005~0008의 작은 실기 모범 예제 4개는 `env_aias_test`의 Python 3.12.8·PyTorch 2.5.1에서 실제 실행해 통과했다. 원본 교안 전체 실행이나 브라우저 UI 검증은 별개다. 교안 pyproject의 torch>=2.9.1 요구와 이 검증 환경 버전은 다르며 패키지는 변경하지 않았다.
- 세 번째 실습 수업은 `lessons/0004-gpt-dataset.html`이다. Dataset 상속, 세 메서드, max_length·stride 시각화, i와 idx의 차이 및 연습 7개를 제공했다. 학습자의 독립 코딩 결과는 아직 확인하지 않았다.
- 두 번째 실습 수업은 `lessons/0003-text-to-next-token.html`이다. 파일 준비·토큰 수·슬라이싱과 shift 시각화, 7개 연습 문제를 제공했다. 제출 전이므로 숙달 여부는 미확인이다.
- 첫 토크나이저 실습 수업은 `lessons/0002-bpe-encode-decode.html`이다. 사용자 요청에 따라 흐름 그림, 짧은 개념 설명, 자동 확인 빈칸·개념 문제, 독립 코딩 과제를 구성했다. 문제 제출은 아직 없으며 숙달로 인정하지 않았다.
- 환경: 기존 Miniconda Python 3.12.8로 `llm_hands_on/.venv`를 만들고 `ipykernel`을 설치했다. 교안의 학습용 패키지 전체는 아직 설치하지 않았다. VS Code 선택 절차는 `PYTHON-SETUP.md`를 참고한다.
- 이후 사용자는 Anaconda Prompt에서 `conda activate env_aias_test`, `cd /d C:\llm_hands_on`, `jupyter lab`으로 실행에 성공했다고 보고했다. 환경은 Python 3.12.8·ipykernel 6.29.5·PyTorch 2.5.1로 확인했다. VS Code도 같은 환경을 쓰도록 기본 경로와 `PYTHON-SETUP.md`를 갱신했다. UI에서 선택된 인터프리터·노트북 커널은 사용자가 직접 확인해야 한다. 기존 `.venv`와 구분한다.
- GitHub 이슈 추적과 triage 라벨 설정은 이전 대화에서 확정되지 않았다. 이번 작업은 학습 공간 구성이다.
