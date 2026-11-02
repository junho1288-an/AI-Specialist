# 지도 메모

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
- 이후 사용자는 Anaconda Prompt에서 `conda activate env_aias_test`, `cd /d C:\llm_hands_on`, `jupyter lab`으로 실행에 성공했다고 보고했다. 현재 안내는 `CMD-JUPYTER-TUTORIAL.md`의 이 절차를 우선한다. 해당 Conda 환경의 Python 버전과 노트북 셀 실행 결과는 아직 확인되지 않았다. 기존 `.venv` 설정과 구분한다.
- GitHub 이슈 추적과 triage 라벨 설정은 이전 대화에서 확정되지 않았다. 이번 작업은 학습 공간 구성이다.
