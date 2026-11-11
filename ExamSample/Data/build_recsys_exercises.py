"""Five offline exercises covering all eight GCF practice markers."""
import json,re,textwrap,difflib
from pathlib import Path
import nbformat as nb
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
SRC=ROOT/'data-class/7_8-recsys-practice'
practice=json.loads((SRC/'RecSys_GCF_practice.ipynb').read_text(encoding='utf-8'))
solution=json.loads((SRC/'RecSys_GCF_sol.ipynb').read_text(encoding='utf-8'))
SETUP='''import torch
import torch.nn as nn
torch.set_num_threads(1)
MISSING=object()  # ???? 옆 MISSING을 정답으로 바꾸세요.

class MessageWeights(nn.Module):
    def __init__(self,dim,out_dim):
        super().__init__()
        self.W1=nn.Linear(dim,out_dim)
        self.W2=nn.Linear(dim,out_dim)

class SmallGraphLayer(nn.Module):
    # 이 문제의 의존성을 없애기 위한 제공 구현입니다.
    # 원본 index_add 결과 유실과 in/out 차원이 다를 때 누적 shape를 보정했습니다.
    def __init__(self,dim,out_dim):
        super().__init__()
        self.W1=nn.Linear(dim,out_dim)
        self.W2=nn.Linear(dim,out_dim)
        self.calls=[]
    def forward(self,edge_index,node_features,user_num,item_num):
        assert node_features.shape[0]==user_num+item_num
        self.calls.append((user_num,item_num))
        src,dst=edge_index
        degree=torch.bincount(edge_index.flatten(),minlength=node_features.shape[0]).float()
        norm=(degree[src]*degree[dst]).rsqrt().unsqueeze(1)
        user_message=(self.W1(node_features[dst])+self.W2(node_features[src]*node_features[dst]))*norm
        item_message=(self.W1(node_features[src])+self.W2(node_features[src]*node_features[dst]))*norm
        result=self.W1(node_features)
        result=result.index_add(0,src,user_message)
        result=result.index_add(0,dst,item_message)
        return torch.nn.functional.leaky_relu(result,negative_slope=.2)

def same(a,e):
    if isinstance(e,torch.Tensor):torch.testing.assert_close(a,e,rtol=1e-5,atol=1e-6)
    elif isinstance(e,(tuple,list)):
        assert len(a)==len(e)
        for x,y in zip(a,e):same(x,y)
    else:assert a==e,(a,e)

def snapshot(v):
    if isinstance(v,torch.Tensor):return v.detach().clone()
    if isinstance(v,(tuple,list)):return tuple(snapshot(x) for x in v)
    return v
'''
spec=[
dict(title='엣지별 degree 정규화',cell=13,count=1,args='deg, src, dst',
 fixture="deg=torch.tensor([2.,1.,1.,2.,0.]) if case==0 else torch.tensor([3.,2.,2.,3.,0.]); src=torch.tensor([0,0,1]); dst=torch.tensor([2,3,3])",
 answer='norm = 1.0 / torch.sqrt(deg[src] * deg[dst])',blank='norm = MISSING',ret='norm',
 explain='norm_e=1/sqrt(deg(user)×deg(item)). src,dst는(E,) 정수 인덱스이고 norm도(E,)입니다. 예: degree2와2가 연결되면 .5. 고립 노드 degree0도 있을 수 있지만 실제 엣지 양 끝의 degree는 양수입니다.'),
dict(title='아이템으로 전달하는 메시지',cell=13,count=2,args='self, src_feat, dst_feat, norm',
 fixture="self=MessageWeights(3,case+2); src_feat=torch.tensor([[1.,2.,-1.],[.5,-.2,1.],[2.,1.,.3]]); dst_feat=torch.tensor([[-1.,.3,2.],[1.,2.,.2],[.4,-1.,2.]]); norm=torch.tensor([.5,.25,.75])",
 answer='edge_messages_for_dst = self.W1(src_feat) + self.W2(src_feat * dst_feat)\nedge_messages_for_dst *= norm.unsqueeze(1)',
 blank='edge_messages_for_dst = MISSING\nedge_messages_for_dst *= MISSING',ret='edge_messages_for_dst',
 explain='item←user 메시지=norm×(W1(h_user)+W2(h_user⊙h_item)). ⊙는 원소별 곱이며 @가 아닙니다. (E,Din)→Linear→(E,Dout),norm(E,)→unsqueeze(1)=(E,1). 예: E3,Dout2이면 각 행을 자기 norm으로 곱합니다.'),
dict(title='아이템 메시지 누적과 자기 정보',cell=13,count=1,args='self, aggregated_messages, dst, edge_messages_for_dst, node_features, user_num',
 fixture="self=MessageWeights(3,3); node_features=torch.randn(5,3); user_num=2; dst=torch.tensor([2,3,3,4]); edge_messages_for_dst=torch.arange(12).float().reshape(4,3)*(case+1); aggregated_messages=torch.zeros_like(node_features); aggregated_messages[:2]=torch.tensor([[1.,2.,3.],[4.,5.,6.]])",
 answer='aggregated_messages.index_add_(0, dst, edge_messages_for_dst)\naggregated_messages[user_num:] += self.W1(node_features[user_num:])',
 blank="raise NotImplementedError('???? dst별 누적과 item 자기 메시지 두 줄을 구현하세요')",ret='aggregated_messages',
 explain='index_add_(0,dst,message)는 같은 dst로 온 메시지를 더합니다. 예: dst=[3,3],메시지[[1,2],[4,5]]→3번 행에[5,7]. 이어 item 부분에 W1(h_item)을 한 번 더합니다. user 행은 그대로 보존합니다. input_dim=output_dim인 원본 조건입니다.'),
dict(title='NGCF 레이어 순서대로 호출',cell=14,count=1,args='self, edge_index',
 fixture="self=nn.Module(); self.num_users=2; self.num_items=3; self.node_embeddings=nn.Embedding(5,3); self.layers=nn.ModuleList([SmallGraphLayer(3,case+2),SmallGraphLayer(case+2,4)]); edge_index=torch.tensor([[0,0,1,1],[2,3,3,4]])",
 answer='node_features=self.node_embeddings.weight\nlayer_outputs=[node_features]\nfor layer in self.layers:\n    node_features=layer(edge_index,node_features,self.num_users,self.num_items)\n    layer_outputs.append(node_features)',
 blank='node_features=self.node_embeddings.weight\nlayer_outputs=[node_features]\nfor layer in self.layers:\n    node_features=MISSING\n    layer_outputs.append(node_features)',ret='layer_outputs',
 explain='처음 임베딩 H0를 보관하고 H1=layer1(edge,H0,U,I),H2=layer2(edge,H1,U,I)를 계산합니다. 다음 레이어에는 갱신한 특징을 넣습니다. 테스트의 제공 레이어는 누적 버그를 보정했고 서로 다른 출력 차원도 지원합니다.'),
dict(title='레이어 특징 결합과 user/item 분리',cell=14,count=3,args='self, layer_outputs',
 fixture="self=nn.Module(); self.num_users=2+case; self.num_items=3; total=self.num_users+self.num_items; layer_outputs=[torch.arange(total*d).float().reshape(total,d)+100*i for i,d in enumerate([2,3,4])]",
 answer='final_features=torch.concat(layer_outputs,dim=-1)\nuser_features=final_features[:self.num_users]\nitem_features=final_features[self.num_users:]',
 blank='final_features=MISSING\nuser_features=MISSING\nitem_features=MISSING',ret='(user_features,item_features)',
 explain='노드 수는 그대로,특징 축을 연결합니다. 예: (5,2)+(5,3)+(5,4)→concat(5,9). U=2이면 users(2,9),items(3,9). node 번호는 user가 앞, item이 뒤입니다. 더하기나 노드축 cat은 다릅니다.')]

def literal(v):
 if isinstance(v,torch.Tensor):return f'torch.tensor({v.tolist()!r},dtype={v.dtype})'
 if isinstance(v,tuple):return '('+', '.join(literal(x) for x in v)+',)'
 return repr(v)

def build():
 found=sum(len(re.findall(r'##\s*fill this part\s*##',''.join(c['source']),re.I)) for c in practice['cells'])
 assert found==sum(s['count'] for s in spec)==8
 cells=[nb.v4.new_markdown_cell('''# Data · NGCF 추천시스템 원본 빈칸

문제: **RecSys_GCF_practice.ipynb** / 정답: **RecSys_GCF_sol.ipynb**.
원본 전체 셀14·15의 빈칸8곳을5문제로 묶었습니다. 문제→빈칸→검증→접힌 해설·정답 순서입니다.

첫 준비 셀을 실행한 뒤 MISSING 또는 NotImplementedError 부분을 작성하세요. 함수 셀→검증 셀 순서로 실행합니다. CPU PyTorch만 필요하고 MovieLens 다운로드·torch-geometric·GPU·전체학습은 필요 없습니다. NGCF는 Neural Graph Collaborative Filtering(신경망 그래프 협업 필터링)입니다.

**원본 제공 코드 보정:** practice와 sol 모두 사용자 누적에서 `aggregated_messages.index_add(...)`의 반환값을 버립니다. index_add는 원본을 바꾸지 않으므로 메시지가 반영되지 않습니다. 이 자료의 제공 레이어는 `result=result.index_add(...)`로 보정했습니다(`index_add_`도 가능). 또한 원본 `zeros_like(node_features)`는 입력/출력 차원이 같을 때만 맞습니다. 제공 레이어는 W1 출력 shape를 사용합니다. 보정은 학습자 빈칸에 추가하지 않았으며 원본 파일도 수정하지 않았습니다.

이 노트북은 빈칸의 작은 그래프 연산을 검사합니다. 실제 추천 정확도·BPR 학습·Recall/NDCG 성능 평가는 포함하지 않습니다. SyntaxError가 나면 수정 후 함수 셀부터 다시 실행하세요.'''),nb.v4.new_code_cell(SETUP)]
 report=[]; mapping=[]
 for i,s in enumerate(spec,1):
  name=f'solve_{i:02d}'
  wrap=lambda b:f'def {name}({s["args"]}):\n'+textwrap.indent(b+'\nreturn '+s['ret'],'    ')
  answer=wrap(s['answer']);blank='import torch\nimport torch.nn as nn\n\n'+wrap('\n'.join(l+'  # ???? 원본 빈칸' if 'MISSING' in l else l for l in s['blank'].splitlines()))
  env={};exec(SETUP,env);exec(answer,env);expected=[]
  for case in range(2):
   torch.manual_seed(4000+i*10+case);env['case']=case;exec(s['fixture'],env)
   expected.append(env['snapshot'](eval(f'{name}({s["args"]})',env)))
  test=f'''def check_{i:02d}():
    expected={literal(tuple(expected))}
    for case in range(2):
        torch.manual_seed({4000+i*10}+case)
'''+textwrap.indent(s['fixture'].replace('; ','\n'),'        ')+f'\n        actual={name}({s["args"]})\n        same(snapshot(actual),expected[case])\n'
  if i==4:test+='        assert all(layer.calls==[(self.num_users,self.num_items)] for layer in self.layers)\n'
  if i==2:test+='        actual.square().mean().backward()\n        assert all(p.grad is not None for p in self.parameters())\n'
  test+=f'''    print('PASS · {i:02d} · 두 그래프/입력 검증')
try:
    check_{i:02d}()
except Exception as error:
    print('FAIL · {i:02d} ·',type(error).__name__,str(error))
'''
  exec(test,env);env[f'check_{i:02d}']()
  exec(blank,env)
  try:env[f'check_{i:02d}']()
  except Exception:pass
  else:raise AssertionError('blank accepted')
  prompt=f'## {i:02d}. {s["title"]}\n\n원본 **전체 셀{s["cell"]+1}** (Markdown 포함,1부터), 빈칸{s["count"]}곳. `self`는 테스트에서 준비한 레이어/모델 객체입니다. 해당 이름을 원본과 같게 유지했습니다. 반환값: `{s["ret"]}`.\n\n입력 예시:\n```python\ncase=0\n{s["fixture"].replace("; ",chr(10))}\n```'
  explanation=f'<details><summary>해설 · 공식 · 정답 보기</summary>\n\n{s["explain"]}\n\n```python\n{answer}\n```\n\n실습 빈칸 밖의 준비는 테스트가 제공합니다. 먼저 시도하고 해설을 확인하세요.\n</details>'
  cells.extend([nb.v4.new_markdown_cell(prompt),nb.v4.new_code_cell(blank),nb.v4.new_code_cell(test),nb.v4.new_markdown_cell(explanation)])
  report.append({'question':i,'source_cell_1based':s['cell']+1,'blanks':s['count'],'cases':2,'answer_pass':True,'blank_rejected':True})
  mapping.append(f'| {i} | {s["cell"]+1} | {s["count"]} | {s["title"]} |')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 nb.validate(book);nb.write(book,OUT/'recsys-gcf-exercises.ipynb')
 (OUT/'recsys-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 diff=''
 for i in [13,14]:diff+='\n'.join(difflib.unified_diff(''.join(practice['cells'][i]['source']).splitlines(),''.join(solution['cells'][i]['source']).splitlines(),fromfile=f'practice cell {i+1}',tofile=f'sol cell {i+1}',lineterm=''))+'\n'
 (OUT/'recsys-source-comparison.diff').write_text(diff,encoding='utf-8')
 (OUT/'RECSYS-README.md').write_text('''# NGCF 추천시스템 원본 빈칸

[문제 노트북](recsys-gcf-exercises.ipynb): 원본 빈칸8곳을5문제로 수록했습니다. `RecSys_GCF_practice.ipynb`가 문제, `RecSys_GCF_sol.ipynb`가 정답입니다. NCF 파일은 요청 범위에 포함하지 않았습니다.

- 준비 셀 실행 후 MISSING/NotImplementedError를 답으로 바꾸고 아래 테스트를 실행하세요.
- CPU PyTorch만 사용합니다. MovieLens·torch-geometric·모델 다운로드는 없습니다.
- 원본 제공 코드의 index_add 반환값 유실을 보정했습니다. index_add는 비인플레이스이므로 반환값을 저장하거나 index_add_를 써야 합니다.
- 원본 zeros_like 누적 버퍼는 입력/출력 차원이 같아야 합니다. 제공 레이어는 W1 출력으로 누적하여 다른 차원도 검사합니다. 원본 빈칸 답 자체는 solution을 따릅니다.
- 같은 아이템으로 향하는 여러 엣지의 합산, user/item 경계, concat 축, 비대칭 입력을 테스트합니다. 실제 추천 품질은 검사하지 않습니다.
- [원본/정답 비교](recsys-source-comparison.diff). 재생성은 `python build_recsys_exercises.py` (nbformat 필요). 학습 답안을 덮어쓰므로 작성 중 실행하지 마세요.

| 문제 | 원본 셀 | 빈칸 | 주제 |
|---|---|---|---|
'''+ '\n'.join(mapping)+'\n',encoding='utf-8')
 print('5 questions / 8 source blanks generated')

if __name__=='__main__':build()
