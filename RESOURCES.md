# PyTorch 실기 학습 자료

## Knowledge

- [Day 4 ResNet 0~8번 줄별 Jupyter 해설](practice/day-04/resnet18-line-by-line.ipynb): 원본 전체 셀 3~21의 코드 셀 11개를 누락 없이 해설했다. 원문과 보충 예제를 구분하고, CPU 예제 9개는 새 커널에서 실행 검증했다. 7~8번은 시험 범위 밖의 추가 이해용이다. 기존 [코드 복습표](reference/day04-resnet-code-map.html)를 요약 자료로 함께 사용한다.

- Day 4 DETR: [매칭·loss 그림 수업](lessons/0029-day04-detr-matching-loss.html), [요약](reference/day04-detr-loss.html). 교안 `04_Applications.pdf` 29~33쪽, 실습 셀 12·18과 공식 [HungarianMatcher](https://github.com/facebookresearch/detr/blob/main/models/matcher.py), [SetCriterion](https://github.com/facebookresearch/detr/blob/main/models/detr.py)를 대조했다. 매칭의 클래스·L1·GIoU 합산 비용, 학습 loss와의 차이, 미배정 no-object 처리를 구분한다.

- Day 4 ResNet 코드 연결 확장: [기존 9단계 수업](lessons/0028-day03-resnet18-shape-journey.html)에 [Vision 모의시험](practice/day-03/vision-mock-exam.ipynb)의 VISION01~05와 residual·stage·pooling 보충 함수를 연결했다. [코드 복습표](reference/day04-resnet-code-map.html). 이론 학습 후 HTML에서 빈칸을 시도하고 Jupyter 실행으로 넘어간다. 원본 실습 셀 6·9·11·15·17과 기존 검증된 모의시험이 근거다.

- Day 3 ResNet shape 수업: `02_DNN_CNN.pdf` 43쪽 전체 텍스트와 실습 셀 6·9·11·13·15를 확인했다. PDF 40쪽의 ResNet 표 이미지를 추출·육안 확인하고 `assets/day03-resnet-slide.png`에 보관했다. [9단계 수업](lessons/0028-day03-resnet18-shape-journey.html)은 노트북 CIFAR 수정 모델을 기준으로 한다. [torchvision 0.20 ResNet 소스](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/resnet.html)와 대조했고 실제 빌더의 B=2 forward shape를 검증했다. 학습된 모델의 인식 성능을 검증한 것은 아니다.

- [Day 3 ResNet18·CIFAR-10 실습](vision/01_ResNet18_CIFAR10.ipynb): 29개 셀 구성·함수 목록을 2026-11-04 확인했다. 전처리·ResNet 구성·학습/평가·추론·성능 분석·activation/kernel/Grad-CAM 시각화와 추가 실습을 포함한다. 전체 실행은 하지 않았다. 이후 질문은 해당 셀 원문을 읽고 설명한다.

- Vision Intro 코드: `01_Intro.pdf` 31쪽 전체 텍스트를 확인했고 Python API는 17쪽에서 식별했다(전 페이지 이미지 OCR·육안 검사는 미수행). [코드 모음](vision/01_Intro-code-snippets.md)에 원문 API 4개와 보충 예제를 구분했다. torchvision 0.20 [ToTensor](https://docs.pytorch.org/vision/0.20/generated/torchvision.transforms.ToTensor.html)·[Normalize](https://docs.pytorch.org/vision/0.20/generated/torchvision.transforms.Normalize.html), Pillow Image, PyTorch unsqueeze 공식 문서를 대조했고 로컬 0.20.1·torch 2.5.1로 계산을 검증했다.

- **Day 3 / Vision 교안:** [교안 목록](vision/README.md). `01_Intro.pdf` 31쪽, `02_DNN_CNN.pdf` 43쪽, `03_Attention.pdf` 59쪽, `04_Applications.pdf` 101쪽. 2026-11-04 각 첫 2쪽과 Intro의 과정 개요를 확인했다. 전체 본문 검토·실습 실행은 아직 하지 않았다. Intro 2쪽에는 DNN·CNN/ResNet → Transformer·ViT → Detection/DETR·Tracking·Segmentation/U-net·생성 모델의 일정이 제시되어 있으며, 전부 오늘 배웠거나 시험에 확정 출제된다는 뜻은 아니다.

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
