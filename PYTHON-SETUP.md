# VS Code에서 Conda 환경으로 Jupyter Notebook 실행하기

[Anaconda Prompt 튜토리얼](CMD-JUPYTER-TUTORIAL.md)의 `env_aias_test`를 VS Code에서도 사용한다. 새 가상환경을 만들 필요가 없다.

이 PC에서 확인한 환경:

```text
C:\ProgramData\miniconda3\envs\env_aias_test\python.exe
Python 3.12.8 / ipykernel 6.29.5 / PyTorch 2.5.1
```

## 1. VS Code 확장 확인

`Ctrl+Shift+X`로 확장 화면을 열고 Microsoft의 **Python**과 **Jupyter**가 설치·활성화되어 있는지 확인한다. 이 PC에서는 두 확장의 설치를 확인했다. 다른 VS Code 프로필이나 원격 창을 사용하면 그 환경에서도 확인한다.

## 2. 교안 폴더 열기

VS Code의 `File → Open Folder`에서 실제 교안 폴더를 연다.

- 이 작업 공간의 교안: `C:\AI-Specialist\llm_hands_on`
- Anaconda Prompt에서 성공한 별도 위치: `C:\llm_hands_on`

둘 중 실제로 작업할 파일이 있는 폴더를 선택한다. 폴더 위치와 Python 환경 위치는 서로 달라도 된다.

Anaconda Prompt에서 열고 싶다면:

```bat
conda activate env_aias_test
cd /d C:\AI-Specialist\llm_hands_on
code .
```

교안이 `C:\llm_hands_on`에 있으면 cd 경로를 그에 맞게 바꾼다. `code .`은 현재 폴더를 VS Code로 여는 명령이다. 활성화한 터미널에서 열어도 아래 인터프리터·커널 선택을 확인한다.

## 3. Python 인터프리터 선택

1. `Ctrl+Shift+P`를 누른다.
2. `Python: Select Interpreter`를 검색한다.
3. `env_aias_test`를 선택한다.
4. 목록에 없으면 `Enter interpreter path` → `Find`에서 다음 파일을 지정한다.

```text
C:\ProgramData\miniconda3\envs\env_aias_test\python.exe
```

`C:\ProgramData\miniconda3\python.exe`는 base 환경이다. 이번에 사용할 것은 그 안의 `envs\env_aias_test\python.exe`다. 다른 PC에서는 `conda env list`로 실제 경로를 확인한다.

## 4. 노트북의 커널도 선택 — 가장 중요한 단계

1. 교안의 `.ipynb` 파일을 연다.
2. 오른쪽 위 **Select Kernel**을 누른다. 이미 커널 이름이 보이면 그 이름을 누른다.
3. **Select Another Kernel → Python Environments**를 선택한다.
4. **env_aias_test**를 선택한다.

명령 팔레트의 `Notebook: Select Notebook Kernel`로도 같은 선택 창을 열 수 있다. 메뉴명은 VS Code 버전에 따라 조금 다를 수 있다.

Python 인터프리터와 노트북 커널은 별도로 선택될 수 있다. `.venv`나 base가 남아 있다면 이 단계에서 바꾼다. 새로 여는 노트북에서도 처음에는 커널을 확인한다.

**VS Code가 로컬 Python 커널을 직접 실행하므로 `jupyter lab`을 따로 켜지 않아도 된다.** 이 방식은 `Existing Jupyter Server` 연결을 선택하는 방식과 구분한다.

## 5. 셀 하나로 실제 환경 확인

새 코드 셀에 입력하고 `Shift+Enter`로 실행한다.

```python
import sys
import torch

print("Python 경로:", sys.executable)
print("Python 버전:", sys.version)
print("PyTorch 버전:", torch.__version__)
print("테스트:", 1 + 2)
```

완료 기준:

- Python 경로가 `C:\ProgramData\miniconda3\envs\env_aias_test\python.exe`
- Python 버전이 `3.12.x`
- PyTorch import 성공
- 테스트 출력이 `3`

예를 들어 경로에 `.venv\Scripts\python.exe`가 나오면 다른 환경이다. 4단계에서 커널을 바꾼다. 위 환경의 패키지 import는 명령줄에서 확인했으며, VS Code UI에서 실제 셀 실행은 이 단계로 직접 확인한다.

## 6. 매일 사용하는 순서

VS Code에서 폴더 열기 → `.ipynb` 열기 → 오른쪽 위 커널 확인 → 위에서부터 셀 실행 → `Ctrl+S` 저장.

셀을 수정하며 실행 순서가 섞였다면 커널 재시작 후 전체 실행으로 확인한다. `????`가 남은 실습 셀은 먼저 빈칸을 채워야 실행된다.

## 커널 목록에 환경이 안 보이면

먼저 `Ctrl+Shift+P` → `Developer: Reload Window`로 VS Code 창을 다시 로드하고 커널 목록을 확인한다. 그래도 없으면 Anaconda Prompt에서 다음으로 환경과 ipykernel을 확인한다.

```bat
conda activate env_aias_test
python -c "import sys, ipykernel; print(sys.executable); print(ipykernel.__version__)"
```

ipykernel이 없다는 오류가 날 때만 같은 환경에서 설치한다.

```bat
python -m pip install ipykernel
```

추가로 명확한 이름의 커널이 필요하면 아래 명령으로 등록할 수 있다. 현재 PC에서는 필수 단계가 아니다.

```bat
python -m ipykernel install --user --name env_aias_test --display-name "Python 3.12 (env_aias_test)"
```

등록 후 VS Code를 다시 로드하고 커널 선택의 `Jupyter Kernels` 등에서 해당 이름을 확인한다.

## 설정 파일의 역할

현재 작업 공간과 그 안의 교안 폴더에는 기본 Python 경로를 다음처럼 설정했다.

```json
{
    "python.defaultInterpreterPath": "C:/ProgramData/miniconda3/envs/env_aias_test/python.exe"
}
```

이 경로는 이 PC 기준이다. 다른 PC에서는 실제 환경 경로로 바꾼다. 기본 경로를 수정해도 VS Code가 이미 저장한 인터프리터나 노트북 커널 선택이 자동 변경된다고 보장되지 않으므로 3~5단계를 확인한다. 별도 폴더 `C:\llm_hands_on`의 설정은 이번 작업에서 수정하지 않았다.

공식 출처: [VS Code Jupyter 커널 선택](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management), [Python 환경 선택](https://code.visualstudio.com/docs/python/environments).
