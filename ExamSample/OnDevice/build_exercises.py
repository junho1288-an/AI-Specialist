"""Build compact, offline exercises from the actual answer regions."""
import json, textwrap, html, ast
from pathlib import Path
import nbformat as nb
import torch

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'2nd_week/Practice_materials'
OUT=Path(__file__).parent

SETUP='''import torch
import torch.nn as nn  # nn.Linear, nn.LayerNorm 등
from collections import namedtuple
from types import SimpleNamespace
torch.set_num_threads(1)
Codebook = namedtuple('Codebook', ['centroids', 'labels'])

def missing():
    raise NotImplementedError('???? 표시의 missing()을 정답 코드로 바꾸세요.')

def pseudo_quantize_tensor(w, n_bit, q_group_size):
    # 테스트용 고정 간격 양자화. 실제 그룹 양자화 구현은 평가하지 않습니다.
    assert n_bit == 3 and q_group_size == 2
    return torch.round(w * 4) / 4

def greedy_step(model, x, cache):
    # 실제 모델/캐시 대신 토큰+1, 호출 횟수+1
    return (x + 1) % 10, cache + 1

class DummyTarget:
    def __call__(self, ids, past_key_values, use_cache):
        assert use_cache
        logits = torch.nn.functional.one_hot((ids+1)%10, 10).float()*3
        return SimpleNamespace(logits=logits, past_key_values=past_key_values+ids.shape[1])

class DummyTree:
    def __call__(self, ids, output_orig, past_key_values, position_ids):
        assert output_orig and torch.equal(position_ids, torch.arange(ids.shape[1]).view(1,-1)+4)
        assert past_key_values == 7
        return {'hidden_states': ids.float()}, torch.nn.functional.one_hot(ids%10,10).float(), ids.float().unsqueeze(-1)

class DummyDraft:
    def compute_logits(self, hidden, output_head):
        return output_head(hidden)

class TinyNet(nn.Module):
    def __init__(self, pair=False):
        super().__init__()
        self.hidden = nn.Linear(4, 5)
        self.head = nn.Linear(5, 3)
        self.pair = pair
    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        y = self.head(h)
        return (y,h) if self.pair else y

def snapshot(x):
    if isinstance(x, torch.Tensor): return x.detach().clone()
    if isinstance(x, nn.Module): return {k:v.detach().clone() for k,v in x.state_dict().items()}
    if isinstance(x, dict): return {k:snapshot(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return tuple(snapshot(v) for v in x)
    return x

def same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), 'Tensor를 반환해야 합니다.'
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys()==expected.keys()
        for k in expected: same(actual[k],expected[k])
    elif isinstance(expected, tuple):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected): same(a,e)
    elif isinstance(expected,float):
        assert abs(actual-expected)<1e-6, (actual,expected)
    else: assert actual==expected, (actual,expected)
'''

# entry: source file, cell, block, title, arguments, fixture, outputs, explanation
spec=[]
def add(f,c,b,title,args,fixture,ret,explain,pre=''):
    spec.append(dict(f=f,c=c,b=b,title=title,args=args,fixture=fixture,ret=ret,explain=explain,pre=pre))

add(1,27,0,'실수 나누기와 반올림','fp_tensor, scale',"fp_tensor=torch.tensor([[-.73,.26,.52],[.11,-.18,.91]])*(case+1); scale=.1*(case+1)",'(scaled_tensor, rounded_tensor)','q=round(r/s)+z. 먼저 r/s를 구하고 torch.round를 적용합니다. 예: .26/.1=2.6 → 3. 입력과 출력 shape=(2,3).')
add(1,27,1,'zero point 이동','rounded_tensor, zero_point',"rounded_tensor=torch.tensor([[-3,0,2],[4,1,-1]],dtype=torch.int32); zero_point=case*3-2",'shifted_tensor','정수 코드에 z를 더합니다. 예: 3+10=13. 여기서는 clamp 이전 단계만 연습합니다.')
add(1,36,0,'scale과 zero point 계산','fp_max, fp_min, quantized_max, quantized_min',"fp_max=1.3+case; fp_min=-.7; quantized_min=-8; quantized_max=7",'(scale, zero_point)','s=(rmax-rmin)/(qmax-qmin), z=round(qmin-rmin/s). 예: 범위[-1,1],[-2,1]이면 s=2/3,z=0. Python 숫자이므로 round 사용. 상수 입력은 제외합니다.')
for cell,shape,ws in [(58,'(2,3)','(1,3)'),(66,'(2,3,2,2)','(1,3,1,1)')]:
 add(1,cell,0,'정수 '+('Linear' if cell==58 else 'Conv2d')+' 출력 재스케일','output, input_scale, weight_scale, output_scale, output_zero_point',f"output=torch.arange({6 if cell==58 else 24}).float().reshape{shape}-3; input_scale=.2; weight_scale=torch.tensor([.1,.3,.5]).reshape{ws}; output_scale=.1*(case+1); output_zero_point=case-2",'output','제공 output은 이미 정수 누적과 bias 보정이 끝난 값입니다. output × (입력 scale × weight scale / 출력 scale) + 출력 z. 예: 10×(.2×.3/.1)+2=8. 채널별 scale의 broadcast를 확인하세요.')
add(1,88,0,'비트 수와 클러스터 개수','bitwidth','bitwidth=case+2','n_clusters','b비트의 번호는 2**b개입니다. 예: 2비트는 00,01,10,11 → 4개. KMeans 학습은 하지 않습니다.')
add(1,88,1,'코드북 디코딩','codebook',"codebook=Codebook(torch.tensor([.2,1.4,-2.])* (case+1),torch.tensor([2,0,2,1,0,1]))",'quantized_tensor','centroids는 대표값, labels는 대표값 번호입니다. 예: [.2,1.4,-2][[2,0]] → [-2,.2]. 결과는 labels와 같은 길이의 float 텐서입니다.')
add(1,99,0,'코드북 중심값 갱신','fp32_tensor, codebook',"fp32_tensor=torch.tensor([.1,1.1,2.4,.5,1.7,2.8])*(case+1); codebook=Codebook(torch.zeros(3),torch.tensor([0,1,2,0,1,2]))",'codebook.centroids','labels==k는 길이6의 bool 텐서입니다. True 위치를 골라 평균냅니다. 예: k=0 → [.1,.5] → .3. 모든 클러스터는 비어 있지 않습니다.',pre='for k in range(codebook.centroids.numel()):')
add(2,30,0,'상위 1% 채널 번호','importance',"importance=torch.arange(100*(case+1)).float().roll(13)",'outlier_mask','topk(x,k)는 (큰 값, 그 위치)를 반환합니다. [1]은 int64 인덱스입니다. 예: 200채널이면 2개 선택, shape=(2,).')
add(2,30,1,'확대한 열 다시 축소','m, outlier_mask, scale_factor',"m=nn.Linear(4,2,bias=False); m.weight.data.copy_(torch.arange(8).view(2,4)+1.); outlier_mask=torch.tensor([1,3]); scale_factor=case+2.",'m','모든 행에서 선택한 열만 scale_factor로 나눕니다. W(2,4), indices(2,) → 선택(2,2). 예: 선택된6을3으로 나누면2. 앞선 확대·양자화는 이미 완료된 상태입니다.')
add(2,37,0,'AWQ 스케일 지수','s_x, ratio',"s_x=torch.tensor([1.,4.,9.]); ratio=[.5,.25][case]",'scales','공식 s=s_x^α, ratio=α. 예: [1,4,9]**.5=[1,2,3]. 뒤의 정규화 이전 값입니다.')
add(2,37,1,'확대 → 양자화 → 축소','fc, scales, w_bit, q_group_size',"fc=nn.Linear(4,2,bias=False); fc.weight.data.copy_(torch.tensor([[.13,.27,.42,.59],[-.31,.62,.78,-.91]])); scales=torch.tensor([1.,2.,3.,4.])*(case+1); w_bit=3; q_group_size=2",'fc','W*s → Q(W*s) → /s. 예: .13×2=.26 → 더미 양자화 .25 → /2=.125. 더미는 .25 간격이고 실제 그룹 양자화 품질은 검사하지 않습니다.')
add(2,53,0,'LayerNorm 스케일 이동','ln, scale',"ln=nn.LayerNorm(3); ln.weight.data.copy_(torch.tensor([2.,3.,4.])); ln.bias.data.copy_(torch.tensor([.3,-.4,.5])); scale=torch.tensor([1.,2.,4.])*(case+1)",'ln','LN(x)=γ*xhat+β. γ와β 모두 s로 나누면 LN출력/s가 됩니다. 예: (2*xhat+4)/2=xhat+2.')
add(2,53,1,'후속 FC 보상','fc, scale',"fc=nn.Linear(3,2); fc.weight.data.copy_(torch.arange(6).reshape(2,3)+1.); scale=torch.tensor([1.,2.,4.])*(case+1)",'fc','입력/s를 보상하려면 weight 입력 열마다 s를 곱합니다. 예: (12/3)×(2×3)=24. bias는 바꾸지 않습니다.')
add(2,58,0,'SmoothQuant 스케일 공식','act_scales, weight_scales, alpha',"act_scales=torch.tensor([1.,4.,9.]); weight_scales=torch.tensor([4.,1.,9.]); alpha=[.5,.25][case]",'scales','s=a^α / w^(1-α). α=.5이면 sqrt(a/w). 예: a=4,w=1 → s=2. 모든 입력 통계는 양수입니다.')
add(2,70,0,'회전 행렬 방향','m, n, R1',"n=['embed','o_proj','q_proj'][case]; m=nn.Embedding(3,2) if case==0 else nn.Linear(3,2,bias=False) if case==1 else nn.Linear(2,3,bias=False); m.weight.data.copy_(torch.arange(6).reshape_as(m.weight).float()); R1=torch.tensor([[0.,-1.],[1.,0.]])",'m','Embedding과 일반 Linear는 W@R, o_proj/down_proj는 R.T@W. shape가 맞는 쪽에서 곱합니다. 회전은 단일 레이어의 동일 출력 자체를 보장하는 것이 아니라 모델의 대응 변환 일부입니다.')
for c,kind,weight in [(26,'soft','soft_target_loss_weight'),(39,'cosine','hidden_rep_loss_weight'),(52,'mse','feature_map_weight')]:
 args='teacher, student, inputs, labels, ce_loss, '+ ('T, soft_target_loss_weight, ce_loss_weight' if kind=='soft' else ('cosine_loss, hidden_rep_loss_weight, ce_loss_weight' if kind=='cosine' else 'mse_loss, feature_map_weight, ce_loss_weight'))
 fixture=f"teacher=TinyNet(pair={kind!='soft'}); student=TinyNet(pair={kind!='soft'}); teacher.eval(); student.train(); inputs=torch.randn(case+2,4); labels=torch.arange(case+2)%3; ce_loss=nn.CrossEntropyLoss(); T=case+2.; {weight}=.3+case*.2; ce_loss_weight=.7-case*.2; cosine_loss=nn.CosineEmbeddingLoss(); mse_loss=nn.MSELoss()"
 explain={'soft':'p=softmax(teacher/T),q=softmax(student/T). soft loss=T²*mean_batch(sum_class p*log(p/q)). 최종=soft가중치×KL+CE가중치×CE. 예: 두 분포가 같으면 KL=0.','cosine':'target=+1이면 hidden 방향이 같아지도록 합니다. 예: [1,0],[2,0]은 cosine loss=0. logits에는 정답 CE를 더합니다. 원본 .cuda()를 .to(inputs.device)로 바꿨습니다.','mse':'학생 feature와 교사 feature의 원소별 제곱차 평균을 CE와 결합합니다. 예: [1,2]와[1,4]의 MSE=2. 더미 feature shape=(B,5).'}[kind]
 add(3,c,0,'지식증류 '+kind,args,fixture,'loss',explain)
add(4,10,0,'KV Cache 바이트 계산','batch_size, n_layers, n_kv_heads, seq_len, head_dim, dtype_bytes',"batch_size=case+1; n_layers=3; n_kv_heads=2; seq_len=5+case; head_dim=4; dtype_bytes=2",'total_bytes','2(K,V)×B×층수×KV헤드×토큰수×헤드차원×원소바이트. 예: B=1,L=3,H=2,S=5,D=4,2bytes → 480bytes. model.config 전역 의존은 제거했습니다.')
add(4,21,0,'H2O 점수','attn_kv, recent_size',"attn_kv=torch.arange(2*3*5*6).float().reshape(2,3,5,6)/100; recent_size=case+1",'scores','Query축(-2)을 합해 (B,H,K)로 만듭니다. 최근 recent_size개의 Key에 inf를 넣습니다. 예: (2,3,5,6) → (2,3,6). recent_size>0인 범위입니다.')
add(4,27,0,'SnapKV 관찰창과 주변 최대','attn_kv, w, K, kernel',"attn_kv=torch.rand(2,3,6,8); w=case+1; K=8; kernel=[3,5][case]",'scores','최근 w개 Query 합산 → Key축 이웃 최대값 → 최근 w개 Key inf. 예: [.2,1,.3,.3,.2],kernel3 → [1,1,1,.3,.3]. kernel은 홀수, w는 양수입니다.')
add(5,9,0,'Magnitude 가지치기 마스크','W, sparsity',"W=torch.tensor([[.1,-.8,.3,.6],[-.2,.9,-.4,.7]]); sparsity=[.25,.5][case]",'mask','abs(W)에서 k번째 작은 값을 기준으로 > 마스크 생성. 예: [.1,.8,.3,.2],50% → [F,T,T,F]. 동점 없는 입력, k>=1입니다. 반환 mask를 W에 곱하는 단계는 원본 제공 코드입니다.')
add(5,15,0,'Wanda 활성값 L2 제곱','x',"x=torch.tensor([[1.,2.,3.],[2.,0.,1.]])*(case+1)",'activation_norm','x=(입력채널,샘플), 행별 sum(x²). 예: [1,2,3] → 1+4+9=14. 전체 합이나 평균과 다릅니다.')
add(5,18,0,'Wanda 행별 마스크','W, sparsity, input_feat, n',"W=torch.tensor([[.1,-.8,.3,.4],[.2,.5,-.6,.1]]); sparsity=[.25,.5][case]; n='linear'; input_feat={n:torch.tensor([10.,1.,2.,1.])}",'mask','importance=abs(W)×입력채널 크기. 각 행에서 k번째 작은 값을 구해 (row,1)로 세워 비교합니다. 예: 첫 행 중요도=[1,.8,.6,.4],50% → [T,T,F,F].')
add(5,27,0,'순차 Draft 생성','model, x, cache, gamma',"model=None; x=torch.tensor([[8-case]]); cache=2; gamma=case+2",'(torch.cat(tokens,dim=-1), cache)','gamma번 greedy_step 호출. 예: 더미 시작8이면 9→0→1. 이전에 생성한 토큰과 갱신한 캐시를 다음 호출에 전달합니다.',pre='tokens = []')
add(5,27,1,'Target에 검증 시퀀스 입력','target, x, drafted, cache',"target=DummyTarget(); x=torch.tensor([[2+case]]); drafted=torch.tensor([[4,5,6]]); cache=7",'(preds, result.past_key_values)','마지막 committed token과 drafted를 길이축으로 cat합니다. 예: [2]+[4,5,6] → [2,4,5,6]. 더미 Target은 각 입력+1을 예측합니다.')
add(5,28,0,'첫 불일치에서 수락 중단','drafted, preds, gamma',"drafted=torch.tensor([[1,2,3]]); preds=[torch.tensor([1,9,3,4]),torch.tensor([1,2,3,4]),torch.tensor([9,2,3,4])][case]; gamma=3",'(new_tokens, n)','앞에서 연속 일치한 토큰만 수락하고 Target 토큰 하나 추가. 예: draft[1,2,3],pred[1,9,3,4] → [1,9],n=1. 뒤의 우연한 일치는 수락하지 않습니다.')
add(5,39,0,'Tree 조상 마스크 상속','tree_mask, mask_index_list',"mask_index_list=[[0,0,1,3],[0,1,0,2]][case]; tree_mask=torch.eye(5,dtype=torch.bool); tree_mask[:,0]=True",'tree_mask','child가 parent의 가시 범위를 OR로 물려받습니다. 예: 0→1→3이면3은0,1,3을 봅니다. 부모가 자식보다 먼저 나오는 순서이며 bool 마스크입니다.')
add(5,41,0,'Tree 전체 한 번에 검증','model, tree_candidates, past_key_values, position_ids',"model=DummyTree(); tree_candidates=torch.tensor([[1,2,3+case]]); past_key_values=7; position_ids=torch.arange(3).view(1,-1)+4",'(outputs, tree_logits, hidden_state)','후보 전체와 캐시·position_ids를 모델에 전달하고 세 출력을 받습니다. 더미가 전달 인자도 검사합니다. EAGLE 설치는 필요 없습니다.')
add(5,42,0,'가장 긴 수락 가지 선택','accept_lengths',"accept_lengths=[torch.tensor([1,3,2]),torch.tensor([2,1,2])][case]",'(best_candidate, accept_length)','argmax는 최대값 위치를 반환합니다. 예: [1,3,2] → 가지1,길이3. 동점이면 첫 위치를 선택합니다.')
add(5,50,0,'병렬 Draft 토큰 선택','draft_hidden, num_draft, model, output_head',"draft_hidden=torch.arange(24).float().reshape(1,6,4)/10; num_draft=case+2; model=DummyDraft(); output_head=nn.Linear(4,7)",'draft_tokens','마지막 num_draft개 hidden을 선택하고 vocabulary logits의 마지막 축 argmax. 예: (1,6,4) → (1,2,4) → (1,2,7) → 토큰(1,2).')

def regions(s):
    blocks=[]; buf=None
    for line in s.splitlines():
        start=('YOUR CODE STARTS HERE' in line or '실습코드 시작' in line or '실습 시작' in line)
        end=('YOUR CODE ENDS HERE' in line or '실습코드 끝' in line or '실습 끝' in line)
        if start: buf=[]
        elif end and buf is not None: blocks.append('\n'.join(buf)); buf=None
        elif buf is not None: buf.append(line)
    return blocks

def literal(x):
    if isinstance(x,torch.Tensor):
        return f'torch.tensor({repr(x.tolist()).replace("inf", "float(\'inf\')")}, dtype={x.dtype})'
    if isinstance(x,dict): return '{'+', '.join(repr(k)+': '+literal(v) for k,v in x.items())+'}'
    if isinstance(x,tuple): return '('+', '.join(literal(v) for v in x)+',)'
    return repr(x)

def build():
    answers={}; students={}; pdfs={}
    for f in range(1,6):
        paths=list(SRC.glob(f'{f}. [ []AVD*')) if False else [p for p in SRC.glob(f'{f}.*ipynb') if '[AVD]' in p.name]
        a=next(p for p in paths if 'answer' in p.name.lower())
        st=next(p for p in paths if not any(w in p.name.lower() for w in ['answer',' my','_my']))
        answers[f]=(a,json.loads(a.read_text(encoding='utf-8')))
        students[f]=st
        pdfs[f]=next(SRC.glob(f'{f}.*slide.pdf'))
    found={(f,i,b) for f,(_,n) in answers.items() for i,c in enumerate(n['cells']) for b,_ in enumerate(regions(''.join(c.get('source',[]))))}
    assert found=={(s['f'],s['c'],s['b']) for s in spec}, (found-{(s['f'],s['c'],s['b']) for s in spec})
    cells=[nb.v4.new_markdown_cell('# On-device 실습 빈칸 모음 · 1~5 통합\n\n모든 문제는 **문제 → 빈칸 코드 → 검증 코드 → 접힌 해설·정답** 순서입니다. PyTorch와 Python 표준 라이브러리만 사용하며 CPU에서 실행합니다. 모델·데이터 다운로드, GPU, transformers, KMeans/EAGLE/DFlash 설치가 필요 없습니다.\n\n첫 준비 셀을 실행하고 원하는 문제의 코드와 테스트를 실행하세요. `????`가 포함된 줄을 완성하세요. 기본 빈칸은 `NotImplementedError`로 표시하여 문법 오류 때문에 이전 함수가 남는 문제를 줄였습니다. 작성 중 SyntaxError가 났다면 먼저 수정하고 문제 셀을 다시 실행해야 합니다. 테스트 PASS는 작은 입력에서의 동작 확인이며 전체 모델 품질·속도 또는 숙련을 의미하지 않습니다.\n\n원본 실습 구간 32개를 독립 함수로 추출했습니다. 바깥 학습 루프·다운로드는 제외하고 필요한 문맥과 변수는 인자로 제공합니다. 원본은 수정하지 않았습니다. 5번의 이미 답이 채워진 표시 구간도 연습용으로 다시 비웠습니다. 각 문제의 테스트 입력은 두 종류 이상입니다. 총 소요 시간은 개인별로 다르므로 한 번에 한 단원씩 진행하세요.'),nb.v4.new_code_cell(SETUP)]
    report=[]; mappings=[]
    env={}; exec(SETUP,env)
    for ix,s in enumerate(spec,1):
        f,c,b=s['f'],s['c'],s['b']; name=f'solve_{ix:02d}'
        block=regions(''.join(answers[f][1]['cells'][c]['source']))[b]
        body=textwrap.dedent('\n'.join(l for l in block.splitlines() if l.strip() and not l.strip().startswith('#')))
        if f==3: body=body.replace('.cuda()', '.to(inputs.device)')
        if (f,c)==(2,58):body='scales = '+body.strip()
        if s['pre']:body=s['pre']+'\n'+textwrap.indent(body,'    ') if s['pre'].endswith(':') else s['pre']+'\n'+body
        solution='@torch.no_grad()\n' if f!=3 else ''
        solution+=f'def {name}({s["args"]}):\n'+textwrap.indent(body+'\nreturn '+s['ret'],'    ')
        exec(solution,env)
        # Explicit arguments and complete context, but keep the required lines blank.
        # Preserve flow/arguments; replace only the exercise operations.
        class Mask(ast.NodeTransformer):
            def visit_Assign(self,node):
                node.value=ast.Call(func=ast.Name(id='missing',ctx=ast.Load()),args=[],keywords=[])
                return node
            def visit_AugAssign(self,node):
                node.value=ast.Call(func=ast.Name(id='missing',ctx=ast.Load()),args=[],keywords=[])
                return node
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call):
                    return ast.Expr(value=ast.Call(func=ast.Name(id='missing',ctx=ast.Load()),args=[],keywords=[]))
                return node
        tree=Mask().visit(ast.parse(body)); ast.fix_missing_locations(tree)
        masked=ast.unparse(tree)
        masked='\n'.join(l+'  # ???? 이 호출을 필요한 식/코드로 바꾸세요.' if 'missing()' in l else l for l in masked.splitlines())
        blank=('import torch\nimport torch.nn as nn\n\n'+('@torch.no_grad()\n' if f!=3 else '')+f'def {name}({s["args"]}):\n'+textwrap.indent(masked+'\nreturn '+s['ret'],'    '))
        ncases=3 if (f,c) in [(2,70),(5,28)] else 2
        expected=[]
        for case in range(ncases):
            env['case']=case; torch.manual_seed(1000+ix*10+case);exec(s['fixture'],env)
            value=eval(f'{name}({s["args"]})',env)
            expected.append(env['snapshot'](value))
            if f==3:
                value.backward();assert all(p.grad is None for p in env['teacher'].parameters());assert any(p.grad is not None and p.grad.abs().sum()>0 for p in env['student'].parameters())
        test=f'''# 작은 CPU 입력으로 검증합니다. 이 셀은 수정하지 마세요.
def check_{ix:02d}():
    expected = {literal(tuple(expected))}
    for case in range({ncases}):
        torch.manual_seed({1000+ix*10}+case)
'''+textwrap.indent(s['fixture'].replace('; ','\n'),'        ')+f'\n        actual = {name}({s["args"]})\n        same(snapshot(actual), expected[case])\n'
        if f==3:test+='        actual.backward()\n        assert all(p.grad is None for p in teacher.parameters()), "교사 gradient가 생성됐습니다."\n        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters()), "학생 gradient를 확인하세요."\n'
        test+=f'''    print("PASS · {ix:02d} · {ncases}개 입력 검증")
try:
    check_{ix:02d}()
except Exception as error:
    print("FAIL · {ix:02d} ·", type(error).__name__, str(error))
'''
        # Run the actual emitted test with the answer; make sure an obvious wrong answer fails.
        exec(test,env)
        exec(f'def {name}({s["args"]}):\n    return None',env)
        try:env[f'check_{ix:02d}']()
        except Exception:pass
        else:raise AssertionError('wrong answer accepted')
        report.append({'question':ix,'source_cell_1based':c+1,'region':b+1,'cases':ncases,'answer_pass':True,'wrong_answer_rejected':True})
        source=students[f].name; ans=answers[f][0].name
        context='\n'.join(l.strip() for l in block.splitlines() if l.strip().startswith('#'))
        prompt=f'## {ix:02d}. {s["title"]}\n\n**원본:** `{source}` 전체 셀 {c+1}, 표시 구간 {b+1}. 답안: `{ans}`. 교안: `{pdfs[f].name}`.\n\n다음 원본 구간의 연산을 구현하세요. 인자 이름은 원본과 같습니다. 출력은 **`{s["ret"]}`** 입니다.\n\n**제공 입력과 shape 확인용 예시** (`case=0`일 때):\n```python\ncase = 0\n{s["fixture"].replace("; ",chr(10))}\n```\n'
        if s['pre']:prompt+=f'\n**필요한 문맥:** 함수 안에서 다음 준비/반복을 포함하세요.\n```python\n{s["pre"]}\n```\n'
        if context:prompt+='\n**원본 힌트**\n```python\n'+context+'\n```\n'
        explanation=f'<details><summary>해설 · 쉬운 공식 · 정답 보기 (먼저 풀어보세요)</summary>\n\n{s["explain"]}\n\n```python\n{solution}\n```\n\n원본 표시 구간을 함수로 분리했습니다. 반환문은 테스트용으로 추가했습니다.\n</details>'
        cells.extend([nb.v4.new_markdown_cell(prompt),nb.v4.new_code_cell(blank),nb.v4.new_code_cell(test),nb.v4.new_markdown_cell(explanation)])
        mappings.append(f'| {ix:02d} | {f} | {c+1} | {b+1} | {s["title"]} |')
    cells[0].source=cells[0].source.replace('기본 빈칸은 `NotImplementedError`로 표시하여', '빈칸은 `missing()` 호출과 `# ????`로 표시합니다. 해당 호출을 정답 식이나 문장으로 바꾸세요. 미완성 상태는 `NotImplementedError`로 알려주어')
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nb.validate(notebook); nb.write(notebook,OUT/'ondevice-exercises.ipynb')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('# On-device 빈칸 모음\n\n[통합 노트북](ondevice-exercises.ipynb)을 열어 준비 셀을 실행하세요. 문제 코드 → 바로 아래 검증 셀 순서로 진행합니다. `NotImplementedError` 줄을 지우고 코드를 작성합니다. 정답은 각 문제 아래 접힌 해설에 있습니다.\n\n- 필요 환경: Python + PyTorch(CPU 가능). 설치·네트워크 호출은 노트북에 없습니다.\n- 원본 1~5번 AVD의 영어/한국어 실습 마커를 전수 대조했습니다. 총 '+str(len(spec))+'개 구간입니다. 다운로드와 긴 외부 실행은 제외했습니다.\n- 원본 바깥 문맥을 함수 인자로 옮기고 테스트용 반환문을 추가했습니다. codebook 중심 갱신은 전체 k 반복을 포함합니다.\n- KD는 학습 루프의 한 배치 손실 부분만 검사하며 teacher gradient 없음과 student gradient 생성을 확인합니다. CPU를 위해 .cuda()를 .to(inputs.device)로 바꿨습니다.\n- KMeans는 군집 알고리즘 대신 클러스터 수·주어진 코드북 복원·평균 갱신을 검사합니다.\n- AWQ의 pseudo quantizer는 고정 .25 간격 더미입니다. 실제 그룹 양자화 품질을 측정하지 않습니다.\n- EAGLE/DFlash/LLM은 인자와 작은 텐서 결과를 검사하는 더미로 대체했습니다. 모델 정확도·처리속도·통합 추론 검사는 아닙니다.\n- pruning은 k>=1, 동점 없는 입력, H2O는 recent_size>0, SnapKV는 양수 window·홀수 kernel 조건입니다.\n- 각 문제 답안을 실제 테스트 셀로 실행하고 None 오답이 거절되는지 확인했습니다. 검증 기록: [validation.json](validation.json).\n- 생성/재검증: `python build_exercises.py` (nbformat 추가 필요). 재생성하면 작성한 답이 덮어써지므로 학습 중 실행하지 마세요.\n\n## 원본 매핑 (셀 번호는 Markdown 포함, 1부터)\n\n| 문제 | 파일 번호 | 셀 | 구간 | 주제 |\n|---|---|---|---|---|\n'+'\n'.join(mappings)+'\n',encoding='utf-8')
    readme=OUT/'README.md'
    readme.write_text(readme.read_text(encoding='utf-8').replace('`NotImplementedError` 줄을 지우고 코드를 작성합니다.', '`# ????`가 붙은 `missing()`을 올바른 식/문장으로 바꿉니다.'),encoding='utf-8')
    print('GENERATED',len(spec),'questions',len(cells),'cells')

if __name__=='__main__':build()
