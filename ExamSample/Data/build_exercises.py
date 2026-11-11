"""Build a compact time-series practice notebook without external data."""
import ast, json, textwrap, re, difflib, contextlib, io
from pathlib import Path
import nbformat as nb
import torch, numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
SRC=ROOT/'data-class/3_4-ts-practice'
sources={p:json.loads((SRC/p).read_text(encoding='utf-8')) for p in ['ts_practice.ipynb','ts_solution.ipynb']}

def source(cell,name=None):
 s=''.join(sources['ts_solution.ipynb']['cells'][cell]['source'])
 if name:
  node=next(n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
  s=ast.get_source_segment(s,node)
 return '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))

SETUP='''import math
import copy
import numpy as np
import torch
import torch.nn as nn  # nn은 torch.nn의 별명입니다.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
torch.set_num_threads(1)
MISSING = object()  # ???? 옆의 MISSING을 정답 코드로 교체하세요.

class TinyRegressor(nn.Module):
    def __init__(self, features=2):
        super().__init__()
        self.fc=nn.Linear(features,1)
        self.calls=[]
    def forward(self,x):
        self.calls.append((self.training,torch.is_grad_enabled()))
        return self.fc(x)

def snapshot(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,np.ndarray):return x.copy()
    if isinstance(x,dict):return {k:snapshot(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return tuple(snapshot(v) for v in x)
    return x

def same(a,e):
    if isinstance(e,torch.Tensor):torch.testing.assert_close(a,e,rtol=2e-5,atol=2e-6)
    elif isinstance(e,np.ndarray):np.testing.assert_allclose(a,e,rtol=1e-6,atol=1e-7)
    elif isinstance(e,dict):
        assert a.keys()==e.keys()
        for k in e:same(a[k],e[k])
    elif isinstance(e,tuple):
        assert len(a)==len(e)
        for v,w in zip(a,e):same(v,w)
    elif isinstance(e,(float,np.floating)):assert abs(a-e)<1e-6,(a,e)
    else:assert a==e,(a,e)
'''
for name,cell in [('EncoderRNN',43),('DecoderRNN',43),('RNNRNN',43),('LSTMModel',21)]:SETUP+='\n\n'+source(cell,name)

spec=[]
def add(title,cell,args,fixture,answer,masked,ret,explain,kind='value',original=True):
 spec.append(dict(title=title,cell=cell,args=args,fixture=fixture,answer=textwrap.dedent(answer).strip(),masked=textwrap.dedent(masked).strip(),ret=ret,explain=explain,kind=kind,original=original))

train='''model.train()
losses=[]
for batch_x,batch_y in batches:
    batch_x,batch_y=batch_x.to(device),batch_y.to(device)
    pred=model(batch_x)[:, -1, 0]
    loss=loss_fn(pred,batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.detach())'''
masked=train.replace('batch_x.to(device),batch_y.to(device)','MISSING').replace('model(batch_x)[:, -1, 0]','MISSING').replace('loss_fn(pred,batch_y)','MISSING').replace('optimizer.zero_grad()','optimizer.MISSING()').replace('loss.backward()','loss.MISSING()').replace('optimizer.step()','optimizer.MISSING()')
add('한 시점 예측: 학습 배치',23,'model, batches, optimizer, loss_fn, device',"model=TinyRegressor(); model.eval(); optimizer=torch.optim.Adam(model.parameters(),lr=.01+case*.01); x=torch.arange(24).float().reshape(2,6,2)/10; batches=[(x,torch.tensor([1.,2.])),(x.flip(1),torch.tensor([2.,1.]))]; loss_fn=nn.MSELoss(); device='cpu'",train,masked,'(losses,model.state_dict(),model.calls)','출력(B,T,1)에서 [:,-1,0]→(B,). MSE→zero_grad→backward→step. 두 배치를 실행해 gradient 누적을 검사합니다. 긴10epoch와 그래프는 제외했습니다.')
evaluation='''model.eval()
losses=[]
with torch.no_grad():
    for test_x,test_y in batches:
        test_x,test_y=test_x.to(device),test_y.to(device)
        test_pred=model(test_x)[:, -1, 0]
        test_loss=loss_fn(test_pred,test_y)
        losses.append(test_loss)
'''
masked=evaluation.replace('test_x.to(device),test_y.to(device)','MISSING').replace('model(test_x)[:, -1, 0]','MISSING').replace('loss_fn(test_pred,test_y)','MISSING')
add('한 시점 예측: 평가 배치',23,'model, batches, loss_fn, device',"model=TinyRegressor(); x=torch.randn(2,4+case,2); batches=[(x,torch.tensor([1.,2.]))]; loss_fn=nn.MSELoss(); device='cpu'",evaluation,masked,'(losses, model.calls)','eval은 Dropout/BN 동작을 바꾸고 no_grad는 gradient 기록을 끕니다. 예: (2,4,1)→마지막 예측(2,).',kind='eval')
metrics='''rmse=np.sqrt(mean_squared_error(y_test,test_predictions))
mape=mean_absolute_percentage_error(y_test,test_predictions)'''
add('RMSE와 MAPE',27,'y_test, test_predictions',"y_test=np.array([2.,4.,8.])*(case+1); test_predictions=np.array([1.,5.,6.])*(case+1)",metrics,metrics.replace('y_test,test_predictions','MISSING, MISSING'),'(rmse,mape)','RMSE=sqrt(mean((y-pred)^2)). MAPE=mean(abs(y-pred)/abs(y)), sklearn은 비율로 반환합니다. .25는25%. MAPE에서 y_true가 첫 인자여야 합니다. 정답0인 입력은 이 테스트에서 제외합니다.')
for name,cell in [('Conv1DModel',29),('RNNModel',34),('EncoderRNN',43),('DecoderRNN',43)]:
 answer=source(cell,name)
 if name=='Conv1DModel':
  masked=answer.replace('in_channels=input_size','in_channels=MISSING').replace('out_channels=hidden_size','out_channels=MISSING').replace('nn.Linear(hidden_size, 1)','nn.Linear(MISSING, 1)')
  args='input_size, hidden_size'; fixture='input_size=case+1; hidden_size=3+case'; ret='Conv1DModel(input_size,hidden_size)'
  explain='Conv1d 입력은(B,F,T). 원본(B,T,F)을 transpose한 뒤 처리합니다. kernel2,stride1,padding0이므로 길이T→T-1. 다시 transpose 후 Linear(H,1).'
 elif name=='RNNModel':
  masked=answer.replace('nn.RNN(input_size, hidden_size, num_layers, batch_first=True)','nn.RNN(MISSING, MISSING, MISSING, batch_first=True)').replace('nn.Linear(hidden_size, 1)','nn.Linear(MISSING, 1)').replace('out, _ = self.rnn(x)\n      return self.fc(out)','return MISSING  # forward 전체를 완성하세요')
  args='input_size, hidden_size, num_layers';fixture='input_size=case+1; hidden_size=3+case; num_layers=case+1';ret='RNNModel(input_size,hidden_size,num_layers)'
  explain='RNN output=(B,T,H),h_n=(layers,B,H). 모든 시점 output을 Linear(H,1)에 전달해(B,T,1)을 반환합니다. output과 최종 hidden을 혼동하지 마세요.'
 else:
  masked=f'''class {name}(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        # 초기화 전체를 완성하세요. super().__init__()도 포함합니다.
        raise NotImplementedError('???? 초기화')
    def forward(self,x{', h' if name=='DecoderRNN' else ''}):
        raise NotImplementedError('???? forward')'''
  args='input_size, hidden_size, num_layers';fixture='input_size=case+1; hidden_size=3+case; num_layers=case+1';ret=f'{name}(input_size,hidden_size,num_layers)'
  explain=('Encoder는 RNN의 마지막 hidden만 반환합니다. (B,T,F)→(layers,B,H). batch_first=True여도 hidden의 순서는 바뀌지 않습니다.' if name=='EncoderRNN' else 'Decoder는 입력x와 전달받은 h를 RNN에 넣고 output을 FC로 투영합니다. 반환(out,h). solution의 FC(H,input_size)를 기준으로 다변량도 검사합니다. copy의 FC(H,1)은 단변량에서만 동일합니다.')
 add(name+' 구성과 forward',cell,args,fixture,answer,masked,ret,explain,kind=name)

OBS='''def observe(actual,kind,case,ctx):
    if kind=='eval':
        losses,calls=actual
        assert all(not training and not grad for training,grad in calls)
        assert all(not v.requires_grad for v in losses)
    if kind in ['Conv1DModel','RNNModel','EncoderRNN','DecoderRNN','LSTMModel','RNNRNN']:
        m=actual
        x=torch.randn(2,5,ctx['input_size'],requires_grad=True)
        if kind=='DecoderRNN':
            h=torch.randn(ctx['num_layers'],2,ctx['hidden_size'],requires_grad=True)
            y=m(x,h)
            assert y[0].shape==(2,5,ctx['input_size'])
        elif kind=='RNNRNN':
            y=m(x,case+2)
            assert y.shape==(2,case+2,ctx['input_size'])
        else:y=m(x)
        if kind=='EncoderRNN':assert y.shape==(ctx['num_layers'],2,ctx['hidden_size'])
        if kind in ['RNNModel','LSTMModel']:assert y.shape==(2,5,1)
        if kind=='Conv1DModel':assert y.shape==(2,4,1)
        loss=sum(v.square().mean() for v in y) if isinstance(y,tuple) else y.square().mean()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum()>0
        grads=tuple((n,p.grad) for n,p in m.named_parameters())
        return snapshot((y,grads))
    return snapshot(actual)
'''
SETUP+='\n\n'+OBS

def literal(x):
 if isinstance(x,torch.Tensor):return f'torch.tensor({x.tolist()!r},dtype={x.dtype})'
 if isinstance(x,np.ndarray):return f'np.array({x.tolist()!r},dtype=np.{x.dtype})'
 if isinstance(x,dict):return '{'+', '.join(repr(k)+':'+literal(v) for k,v in x.items())+'}'
 if isinstance(x,tuple):return '('+', '.join(literal(v) for v in x)+',)'
 if isinstance(x,np.generic):return repr(x.item())
 return repr(x)

def build():
 diffs=[23,27,29,34,43]
 # Verify the solution has no incomplete Python code or exercise markers.
 for c in sources['ts_solution.ipynb']['cells']:
  if c['cell_type']=='code':ast.parse(''.join(c['source']))
 cells=[nb.v4.new_markdown_cell('Data'),nb.v4.new_code_cell(SETUP)]
 rows=[];report=[]
 for ix,s in enumerate(spec,1):
  name=f'solve_{ix:02d}'
  wrap=lambda body:f'def {name}({s["args"]}):\n'+textwrap.indent(body+'\nreturn '+s['ret'],'    ')
  sol=wrap(s['answer']); blank='import torch\nimport torch.nn as nn\n\n'+wrap('\n'.join(l+'  # ???? 빈칸' if 'MISSING' in l and not l.rstrip().endswith(chr(92)) else l for l in s['masked'].splitlines()))
  compile(blank,'blank','exec')
  env={};exec(SETUP,env);exec(sol,env);expected=[]
  for case in range(2):
   torch.manual_seed(3000+ix*10+case);env['case']=case;exec(s['fixture'],env)
   actual=eval(f'{name}({s["args"]})',env)
   expected.append(env['observe'](actual,s['kind'],case,env))
  test=f'''# 이 검증 셀은 수정하지 마세요.
def check_{ix:02d}():
    expected={literal(tuple(expected))}
    for case in range(2):
        torch.manual_seed({3000+ix*10}+case)
'''+textwrap.indent(s['fixture'].replace('; ','\n'),'        ')+f'\n        actual={name}({s["args"]})\n        same(observe(actual,{s["kind"]!r},case,locals()),expected[case])\n'
  test+=f'''    print('PASS · {ix:02d} · 두 입력 검증')
try:
    check_{ix:02d}()
except Exception as error:
    print('FAIL · {ix:02d} ·',type(error).__name__,str(error))
'''
  exec(test,env);env[f'check_{ix:02d}']()
  exec(blank,env)
  try:env[f'check_{ix:02d}']()
  except Exception:pass
  else:raise AssertionError(('blank accepted',ix))
  label='원본 빈칸' if s['original'] else 'solution 기반 추가 빈칸'
  prompt=f'## {ix:02d}. {s["title"]}\n\n**구분: {label}**. 원본 전체 셀 **{s["cell"]+1}** (Markdown 포함,1부터). 원본 빈칸은 ts_practice와 copy 비교, 참고 답안은 ts_solution입니다.\n\n목표: 제공 문맥에서 빈칸을 구현하세요. 반환값은 `{s["ret"]}`입니다.\n\n**입력 예시:**\n```python\ncase=0\n{s["fixture"].replace("; ",chr(10))}\n```'
  explanation=f'<details><summary>해설 · 공식 · 정답 보기 (먼저 풀어보세요)</summary>\n\n{s["explain"]}\n\n```python\n{sol}\n```\n\n테스트용 함수 인자·반환값을 추가하고 긴 실행·다운로드·그래프는 제외했습니다.\n</details>'
  cells.extend([nb.v4.new_markdown_cell(prompt),nb.v4.new_code_cell(blank),nb.v4.new_code_cell(test),nb.v4.new_markdown_cell(explanation)])
  rows.append(f'| {ix:02d} | {label} | {s["cell"]+1} | {s["title"]} |')
  report.append({'question':ix,'original_blank':s['original'],'source_cell_1based':s['cell']+1,'reference_pass':True,'blank_rejected':True,'cases':2})
 cells[0].source='''# Data · 시계열 원본 빈칸 7문제

문제 원본은 **ts_practice.ipynb**, 정답 기준은 **ts_solution.ipynb**입니다. copy 파일은 정답 근거로 사용하지 않습니다. 원본의 미완성5개 셀을 학습/평가, 지표, Conv1D, RNN, Encoder/Decoder의7문제로 묶었습니다. 추가로 만든 문제는 제외했습니다.

**문제 → 빈칸 코드 → 검증 코드 → 접힌 해설·정답** 순서입니다. 첫 준비 셀을 실행하고 MISSING 또는 NotImplementedError 부분을 완성하세요. 문제 셀→검증 셀 순서로 실행합니다. SyntaxError가 나면 수정 후 문제 셀부터 재실행하세요.

CPU PyTorch, NumPy, scikit-learn만 필요합니다. 주가 데이터·인터넷·GPU 없이 작은 더미 시계열로 검사합니다. 긴10epoch 학습·그래프는 제외했습니다. PASS는 작은 입력의 검증 결과이며 실제 예측 품질·숙련을 뜻하지 않습니다.

Decoder의 FC는 정답 파일과 같은 nn.Linear(hidden_size,input_size)입니다. 원본 파일은 수정하지 않았습니다.'''
 for cell in cells:
  if cell.cell_type=='markdown':
   cell.source=cell.source.replace('원본 빈칸은 ts_practice와 copy 비교, 참고 답안은 ts_solution입니다.','문제는 ts_practice.ipynb, 정답은 ts_solution.ipynb를 기준으로 합니다.')
   cell.source=cell.source.replace('copy의 FC(H,1)은 단변량에서만 동일합니다.','정답 파일의 input_size를 그대로 사용합니다.')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 nb.validate(book);nb.write(book,OUT/'timeseries-exercises.ipynb')
 (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 comparison=''
 for i in diffs:comparison+='\n'.join(difflib.unified_diff(''.join(sources['ts_practice.ipynb']['cells'][i]['source']).splitlines(),''.join(sources['ts_solution.ipynb']['cells'][i]['source']).splitlines(),fromfile=f'ts_practice cell {i+1}',tofile=f'ts_solution cell {i+1}',lineterm=''))+'\n'
 (OUT/'source-comparison.diff').write_text(comparison,encoding='utf-8')
 (OUT/'README.md').write_text('''# Data 시계열 원본 빈칸 모음

[통합 노트북](timeseries-exercises.ipynb)을 열어 준비 셀 → 문제 코드 → 검증 코드 순서로 실행하세요. MISSING 또는 NotImplementedError 부분을 완성합니다. 정답은 문제 아래 접혀 있습니다.

- 문제: `data-class/3_4-ts-practice/ts_practice.ipynb`
- 정답: `data-class/3_4-ts-practice/ts_solution.ipynb`
- copy 파일은 정답 기준에서 제외했습니다. solution에 새로 만들었던 추가 문제도 제외했습니다.
- 원본의 미완성 셀24,28,30,35,44를7문제로 구성했습니다. [문제/정답 비교](source-comparison.diff).
- CPU PyTorch, NumPy, scikit-learn만 사용합니다. 데이터 다운로드·GPU·긴학습·그래프는 없습니다.
- Decoder FC는 정답 파일의 `(hidden_size,input_size)`를 따르며 단변량·다변량을 검사합니다.
- MAPE의 .1은10%입니다. 정답0인 샘플은 이 최소 테스트에서 제외합니다.
- 각 문제를 두 입력으로 검증합니다. 모델 출력·shape·gradient·두 배치 업데이트를 검사하며 실제 예측 품질 평가는 아닙니다.
- 재검증: `python validate_exercises.py` (nbformat, nbclient 필요).
- 재생성: `python build_exercises.py`. 학습 중 실행하면 작성한 답을 덮어쓰니 주의하세요.

| 문제 | 구분 | 원본 전체 셀 | 주제 |
|---|---|---|---|
'''+ '\n'.join(rows)+'\n',encoding='utf-8')
 print('Generated',len(spec),'questions')

if __name__=='__main__':build()
