# VS Code 노트북 시작하기

## 준비된 환경

- `.python-version` 요구 버전: Python 3.12
- 사용한 기존 Python: Miniconda Python 3.12.8
- 프로젝트 가상환경: `C:\AI-Specialist\llm_hands_on\.venv`
- 실행 파일: `C:\AI-Specialist\llm_hands_on\.venv\Scripts\python.exe`
- 노트북 실행용 `ipykernel`을 가상환경에 설치했다.
- 루트와 교안 폴더의 VS Code 설정에 기본 Python 경로를 추가했다. 이미 선택된 인터프리터나 노트북 커널은 아래 절차로 직접 변경한다.

## VS Code에서 선택하기

1. VS Code에서 `C:\AI-Specialist` 폴더를 연다.
2. `Ctrl+Shift+P` → `Developer: Reload Window`로 창을 새로고침한다.
3. `Ctrl+Shift+P` → `Python: Select Interpreter` → `Enter interpreter path`에서 위 실행 파일을 선택한다.
4. 교안의 `.ipynb` 파일을 연다.
5. 오른쪽 위 `Select Kernel` → `Select Another Kernel` → `Python Environments`에서 해당 `.venv`를 선택한다. 메뉴 표시는 VS Code 버전에 따라 조금 다를 수 있다.
6. 첫 셀에서 아래 코드를 실행해 경로와 버전을 확인한다.

```python
import sys
print(sys.version)
print(sys.executable)
```

버전은 `3.12.8`, 경로는 `llm_hands_on\.venv\Scripts\python.exe`가 나와야 한다. 셀 실행은 `Shift+Enter`다.

## 설치 범위

이번에는 Python 코드 셀 실행에 필요한 커널을 준비했다. `torch`, `transformers`, `unsloth` 등 `pyproject.toml`의 수업용 패키지 전체는 아직 설치하지 않았다. 각 실습 실행에 필요한 패키지는 하드웨어와 교안 요구사항을 확인해 같은 가상환경에 설치한다.

VS Code 안에서 셀을 실행하는 데 별도의 브라우저 Jupyter 서버를 띄울 필요는 없다. `run_jupyter.bat`의 `jupyter lab`을 사용하려면 JupyterLab을 별도로 설치해야 한다.

## PowerShell에서 같은 환경 사용하기

활성화 없이도 실행 파일 경로를 지정할 수 있다.

```powershell
cd C:\AI-Specialist\llm_hands_on
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

출처: [VS Code 공식 노트북 안내](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
