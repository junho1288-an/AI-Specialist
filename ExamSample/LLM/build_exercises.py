"""Extract the six requested LLM notebooks into offline, independently checked tasks."""
import ast, json, re, textwrap, copy
from pathlib import Path
import nbformat as nb
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
FILES={
 '2':'Chapter_2_Exercise_Dataset.ipynb',
 '3':'Chapter_3_Excercise_Attention.ipynb',
 '4':'Chapter_4_Excercise_GPT.ipynb',
 '5':'Chapter_5_Excercise_Pretraining.ipynb',
 '6L':'Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb',
 '6C':'Chapter_6_Excercise_Finetuning_Classification.ipynb',
}
RAW={k:json.loads((ROOT/'llm_hands_on'/v).read_text(encoding='utf-8')) for k,v in FILES.items()}
ANS={
 ('2',0):['encode','decode'], ('2',1):['1','1'],
 ('2',2):['i','i','i','1','i','1','idx','idx'],('2',3):['dataset'],
 ('2',5):['vocab_size','output_dim','input_ids'],
 ('3',0):['Linear','d_in','d_out']*3+['W_key','x','W_query','x','W_value','x','queries','keys','-torch.inf','softmax','values'],
 ('3',1):['ca'],
 ('4',2):['mean','var','scale','shift','4','GELU','4'],
 ('4',3):['att','ff','vocab_size','context_length','vocab_size','pos_embeds','trf_blocks','out_head'],
 ('4',4):['no_grad','-1','argmax','cat'],
 ('5',1):['cross_entropy','flatten','flatten'],
 ('5',2):['train','zero_grad','calc_loss_batch','backward','step'],
 ('5',3):['-1','argmax','idx_next'],
 ('5',4):['unsqueeze','squeeze','parameters','lr'],
 ('6L',2):['rank','rank','alpha','A','B','lora','Linear'],
 ('6L',3):['False','replace_linear_with_lora'],
 ('6C',0):['self.max_length','pad_token_id','Label','-1'],
 ('6C',1):['False','num_classes','-1','True','True'],
}
SOL={}; LOC={}
for key,replacements in ANS.items():
 lines=''.join(RAW[key[0]]['cells'][key[1]]['source']).splitlines(); it=iter(replacements);loc=[]
 for i,line in enumerate(lines):
  if not line.lstrip().startswith('#'):
   for m in re.finditer(r'\?{3,}',line):loc.append((i,m.start()))
   lines[i]=re.sub(r'\?{3,}',lambda m:next(it),line)
 assert len(loc)==len(replacements),(key,len(loc),len(replacements))
 SOL[key]='\n'.join(lines); LOC[key]=loc

def clean(s):
 s=textwrap.dedent(s)
 s=re.sub(r'(?s)""".*?"""','',s)
 return '\n'.join(l for l in s.splitlines() if l.strip() and not l.lstrip().startswith(('#','print(')))

def segment(f,c,node=None,span=None):
 source=SOL.get((f,c),''.join(RAW[f]['cells'][c]['source']))
 if node:
  obj=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==node)
  start,end=obj.lineno-1,obj.end_lineno
 else:start,end=span
 raw='\n'.join(''.join(RAW[f]['cells'][c]['source']).splitlines()[start:end])
 answer='\n'.join(source.splitlines()[start:end])
 return clean(raw),clean(answer),(start,end)

SETUP='''import math
import copy
import torch
import torch.nn as nn  # torch.nn의 별명
from torch.utils.data import Dataset, DataLoader, TensorDataset
from types import SimpleNamespace
torch.set_num_threads(1)
MISSING = object()  # ???? 위치의 MISSING을 정답으로 교체하세요.

class DummyTokenizer:
    # UTF-8 byte를 번호로 쓰는 오프라인 더미입니다. 실제 GPT-2 BPE 결과가 아닙니다.
    def encode(self, text, allowed_special=None):
        return list(text.encode('utf-8'))
    def decode(self, ids):
        return bytes(ids).decode('utf-8')

class TinyLM(nn.Module):
    def __init__(self, vocab=11, dim=4):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim)
        self.head=nn.Linear(dim,vocab)
    def forward(self,x): return self.head(self.emb(x))

class NextTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(()))
        self.calls=[]
    def forward(self,idx):
        self.calls.append((idx.detach().clone(),torch.is_grad_enabled()))
        return torch.nn.functional.one_hot((idx+1)%7,7).float()*4 + self.anchor

class TinyClassifier(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.emb=nn.Embedding(11,dim)
        self.trf_blocks=nn.Sequential(nn.Linear(dim,dim),nn.Linear(dim,dim))
        self.final_norm=nn.LayerNorm(dim)
        self.out_head=nn.Linear(dim,11)
    def forward(self,x):
        return self.out_head(self.final_norm(self.trf_blocks(self.emb(x))))

class MiniFrame:
    # pandas의 iloc[index]['Label'] 인터페이스만 제공하는 더미입니다.
    def __init__(self, rows): self.iloc=rows

def flags(model): return {n:p.requires_grad for n,p in model.named_parameters()}

def snapshot(x):
    if isinstance(x,torch.Tensor): return x.detach().clone()
    if isinstance(x,dict):return {k:snapshot(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return tuple(snapshot(v) for v in x)
    return x

def same(a,e):
    if isinstance(e,torch.Tensor):
        assert isinstance(a,torch.Tensor), 'Tensor를 반환하세요.'
        torch.testing.assert_close(a,e,rtol=2e-5,atol=2e-6)
    elif isinstance(e,dict):
        assert a.keys()==e.keys()
        for k in e:same(a[k],e[k])
    elif isinstance(e,tuple):
        assert len(a)==len(e)
        for x,y in zip(a,e):same(x,y)
    elif isinstance(e,float):assert abs(a-e)<1e-6,(a,e)
    else:assert a==e,(a,e)
'''
# Supplied helpers are separate from learner functions, so questions are independent.
for f,c,name in [('2',2,'GPTDatasetV1'),('3',0,'CausalAttention'),('4',1,'MultiHeadAttention'),('4',2,'LayerNorm'),('4',2,'GELU'),('4',2,'FeedForward'),('4',3,'TransformerBlock'),('5',1,'calc_loss_batch'),('6L',2,'LoRALayer'),('6L',2,'LinearWithLoRA'),('6L',2,'replace_linear_with_lora')]:
 SETUP+='\n\n'+segment(f,c,node=name)[1]

spec=[]
def add(f,c,title,args,fixture,ret,explain,node=None,span=None,pre='',post='',check='',kind='value',adapt=None):
 raw,answer,bounds=segment(f,c,node,span)
 if adapt:
  for a,b in adapt:raw=raw.replace(a,b);answer=answer.replace(a,b)
 spec.append(dict(f=f,c=c,title=title,args=args,fixture=fixture,ret=ret,explain=explain,raw=raw,answer=answer,bounds=bounds,pre=pre,post=post,check=check,kind=kind))

add('2',0,'토큰 인코딩·디코딩','tokenizer, text',"tokenizer=DummyTokenizer(); text=['Hello, tea?', 'AI 공부'][case]",'(integers, strings)','encode는 문자열→정수 목록, decode는 정수 목록→문자열입니다. 예: 더미에서 A→[65]→A. 실제 GPT-2 BPE 토큰 값과는 다릅니다.',span=(15,24))
add('2',1,'다음 토큰 정답 한 칸 이동','enc_sample, context_size',"enc_sample=list(range(10,22)); context_size=case+3",'(x,y)','입력 [a,b,c]의 정답은 [b,c,d]. 끝 인덱스는 포함하지 않으므로 시작·끝 모두1씩 이동합니다.',span=(29,31))
add('2',2,'GPTDataset 슬라이딩 윈도우','txt, tokenizer, max_length, stride',"txt='abcdefghijk'; tokenizer=DummyTokenizer(); max_length=case+3; stride=case+1",'tuple(GPTDatasetV1(txt, tokenizer, max_length, stride))','입력 token_ids[i:i+L],정답 token_ids[i+1:i+L+1]. 각 샘플은 long 텐서 두 개, shape=(L,). stride로 시작 위치를 이동합니다.',node='GPTDatasetV1')
add('2',3,'DataLoader에 Dataset 전달','dataset, batch_size, shuffle, drop_last, num_workers',"dataset=TensorDataset(torch.arange(15).reshape(5,3),torch.arange(5)); batch_size=case+2; shuffle=False; drop_last=bool(case); num_workers=0",'tuple(dataloader)','DataLoader의 첫 인자는 Dataset입니다. 5샘플,batch2,drop_last=False → 배치크기2,2,1. 문제에서는 토큰화·파일 준비를 생략합니다.',span=(17,26))
add('2',5,'토큰 임베딩 조회','vocab_size, output_dim, input_ids',"vocab_size=8+case; output_dim=3+case; input_ids=torch.tensor([2,0,5,2])",'(embedding_layer.weight, token_embeddings)','Embedding(V,D)는 (V,D) 표입니다. ID4개를 넣으면 (4,D), 같은 ID는 같은 행을 꺼냅니다.',span=(10,20))
add('3',0,'CausalAttention 전체 흐름','d_in, d_out, context_length, qkv_bias',"d_in=3+case; d_out=2+case; context_length=5; qkv_bias=bool(case)",'CausalAttention(d_in,d_out,context_length,0.0,qkv_bias)','Q,K,V는(B,T,Dout), QKᵀ는(B,T,T). 미래점수=-inf → softmax 확률0 → AV는(B,T,Dout). dropout=0으로 검사합니다.',node='CausalAttention',kind='attention')
add('3',1,'Attention 인스턴스 호출','ca, batch',"ca=CausalAttention(3,2,6,0.0); batch=torch.randn(case+1,4,3)",'context_vecs','ca(batch)는 __call__을 통해 forward를 실행합니다. (B,4,3) → (B,4,2).',span=(27,28))
add('4',2,'LayerNorm 평균·분산·학습 파라미터','emb_dim',"emb_dim=case+3",'LayerNorm(emb_dim)','각 토큰의 마지막 축에서 (x-mean)/sqrt(var+eps), 이후 scale×정규화+shift. 예: [1,3]의 평균2,분산1. 학습 scale/shift도 검사합니다.',node='LayerNorm',kind='norm')
add('4',2,'FFN 확장과 GELU','cfg',"cfg={'emb_dim':case+3}",'FeedForward(cfg)','D→4D→GELU→D. 예: (2,3,4)→(2,3,16)→(2,3,4). 원본의 tanh 근사 GELU가 준비 셀에 제공됩니다.',node='FeedForward',kind='ff')
cfg="cfg={'emb_dim':4+case*2,'vocab_size':9+case,'context_length':6,'n_heads':2,'drop_rate':0.,'qkv_bias':True,'n_layers':2}"
add('4',3,'Transformer 잔차 두 경로','cfg',cfg,'TransformerBlock(cfg)','LN→Attention→잔차합, LN→FFN→잔차합. 두 경로 모두 입력 shape 유지. MultiHeadAttention은 원본의 제공 코드를 사용합니다.',node='TransformerBlock',kind='block')
add('4',3,'GPT 임베딩·블록·출력층 조립','cfg',cfg,'GPTModel(cfg)','ID(B,T)→토큰임베딩(B,T,D)+위치(T,D)→블록→최종LN→logits(B,T,V). V는 단어장,context_length는 최대 위치 수입니다.',node='GPTModel',kind='gpt')
add('4',4,'Greedy 생성 루프','model, idx, max_new_tokens, context_size',"model=NextTokenModel(); idx=torch.tensor([[1,2,3],[2,3,4]]); max_new_tokens=case+2; context_size=case+2",'generate_text_simple(model, idx, max_new_tokens, context_size)','마지막 위치 logits에서 argmax(keepdim=True)로(B,1)을 얻고 시퀀스축 cat. 예: [1,2,3]→[1,2,3,4]. no_grad와 context 자르기도 검사합니다.',node='generate_text_simple',check="assert all(not grad and ids.shape[1]<=context_size for ids,grad in model.calls)")
add('5',1,'사전학습 token loss','input_batch, target_batch, model, device',"model=TinyLM(); input_batch=torch.tensor([[1,2,3],[2,4,5]])[:case+1]; target_batch=(input_batch+1)%11; device='cpu'",'calc_loss_batch(input_batch,target_batch,model,device)','logits(B,T,V)→(B*T,V),정답(B,T)→(B*T,). 모든 토큰을 CE에 전달합니다. 예: B2,T3이면 정답6개.',node='calc_loss_batch',kind='loss')
# Keep only the marked training actions, not full epoch/evaluation/generation scaffolding.
trainraw='''model.????()
for input_batch, target_batch in batches:
    optimizer.????()
    loss = ????(input_batch, target_batch, model, device)
    loss.????()
    optimizer.????()
'''
add('5',2,'학습 모드·gradient·업데이트','model, optimizer, batches, device',"model=TinyLM(); model.eval(); optimizer=torch.optim.SGD(model.parameters(),lr=.03+case*.02); x=torch.tensor([[1,2,3],[3,2,1]]); batches=[(x,(x+1)%11),(x.flip(1),(x.flip(1)+2)%11)]; device='cpu'",'(model.state_dict(), model.training)','train→zero_grad→loss→backward→step. 두 배치를 연속 처리해 gradient 초기화 누락도 확인합니다. 모델·optimizer는 테스트가 준비하며 전체 epoch는 제외합니다.',node='train_model_simple')
spec[-1]['raw']=trainraw;spec[-1]['answer']=trainraw
for ans in ['train','zero_grad','calc_loss_batch','backward','step']:spec[-1]['answer']=re.sub(r'\?{3,}',ans,spec[-1]['answer'],count=1)
add('5',3,'EOS가 있는 Greedy 생성','model, idx, max_new_tokens, context_size, eos_id',"model=NextTokenModel(); idx=torch.tensor([[1,2]]); max_new_tokens=5; context_size=3; eos_id=[None,4][case]",'generate(model,idx,max_new_tokens,context_size,eos_id)','원본은 주석과 달리 실제로 greedy만 구현합니다. 마지막 토큰을 뽑고 EOS이면 붙이기 전에 종료합니다. B=1로 제한: 원본 if문은 다중 배치에 맞지 않습니다.',node='generate')
add('5',4,'텍스트에 배치 차원 추가','text, tokenizer',"text=['abc','Hello'][case]; tokenizer=DummyTokenizer()",'text_to_token_ids(text,tokenizer)','encode 결과 길이T → tensor(T,) → unsqueeze(0)=(1,T). 예: [65,66] → [[65,66]].',node='text_to_token_ids')
add('5',4,'토큰에서 배치 차원 제거','token_ids, tokenizer',"tokenizer=DummyTokenizer(); token_ids=torch.tensor([tokenizer.encode(['abc','Hello'][case])])",'token_ids_to_text(token_ids,tokenizer)','(1,T)→squeeze(0)→(T,)→tolist→decode. B=1이라는 전제입니다.',node='token_ids_to_text')
add('5',4,'AdamW 파라미터와 학습률','model, settings',"model=TinyLM(); settings={'learning_rate':.002*(case+1),'weight_decay':.1+case*.1}",'optimizer','AdamW(model.parameters(),lr=...,weight_decay=...). 모든 학습 파라미터가 전달됐는지와 설정을 검사합니다.',span=(141,144),kind='optimizer')
add('6C',0,'분류 데이터 자르기와 패딩','self, pad_token_id',"self=SimpleNamespace(max_length=case+3,encoded_texts=[[1,2],[3,4,5,6,7],[]]); pad_token_id=8+case",'self.encoded_texts','최대길이로 먼저 자르고 부족한 만큼 pad를 붙입니다. 예: max3,pad9이면 [1]→[1,9,9], [1,2,3,4]→[1,2,3]. CSV 다운로드는 제외했습니다.',span=(108,121))
for field in ['raw','answer']:
 spec[-1][field]='\n'.join(l.lstrip() for l in spec[-1][field].splitlines())
add('6C',0,'분류 정답 열 선택','self, index',"self=SimpleNamespace(data=MiniFrame([{'Label':0,'Other':8},{'Label':1,'Other':9}])); index=case",'label','iloc[index]로 해당 행을 찾고 Label 열에서 정답을 얻습니다. 예: index1→Label1. pandas 설치 없이 동일 인터페이스 더미를 제공합니다.',span=(126,127))
add('6C',0,'분류는 마지막 위치 logits','input_batch, target_batch, model, device',"model=TinyLM(vocab=3); input_batch=torch.tensor([[0,1,2],[1,2,0]])[:case+1]; target_batch=torch.tensor([1,0])[:case+1]; device='cpu'",'calc_loss_batch(input_batch,target_batch,model,device)','분류 logits(B,T,C)에서 [:,-1,:]→(B,C). 정답은(B,). 사전학습처럼 모든 토큰을 flatten하지 않습니다.',node='calc_loss_batch',kind='loss')
add('6C',1,'분류 미세조정: 동결·출력층·일부 해제','model, BASE_CONFIG, num_classes',"model=TinyClassifier(); BASE_CONFIG={'emb_dim':4}; num_classes=case+2",'(model, flags(model))','전체 동결 후 새 분류 head를 만들고 마지막 Transformer와 final_norm만 해제합니다. 새 head는 기본 requires_grad=True. 예: 이전 블록False,마지막블록True.',span=(143,171),kind='classification',adapt=[('num_classes = 2',''),('model.to(device)',''),('torch.manual_seed(123)','')])
add('6L',2,'LoRA A·B와 저랭크 곱','in_dim, out_dim, rank, alpha',"in_dim=3+case; out_dim=5; rank=case+1; alpha=3.+case",'LoRALayer(in_dim,out_dim,rank,alpha)','A=(in,rank),B=(rank,out). Δy=(alpha/rank)*x@A@B. 초기B=0이면 Δy=0. 검증에서는 B를0이 아닌 값으로 바꿔 수식과 A/B gradient도 확인합니다.',node='LoRALayer',kind='lora')
add('6L',2,'기존 Linear에 LoRA 출력 더하기','linear, rank, alpha',"linear=nn.Linear(3,4); rank=case+1; alpha=3.",'LinearWithLoRA(linear,rank,alpha)','y=linear(x)+lora(x). 두 출력 모두(B,out)이라 더할 수 있습니다. B를0이 아닌 값으로 바꿔 잔차 누락을 검사합니다.',node='LinearWithLoRA',kind='wrapped')
add('6L',2,'중첩 Linear를 재귀 교체','model, rank, alpha',"model=nn.Sequential(nn.Linear(3,4),nn.Sequential(nn.ReLU(),nn.Linear(4,2))); rank=case+1; alpha=2.",'model','named_children으로 자식 순회. Linear이면 wrapper로 교체하고, 아니면 안쪽으로 재귀합니다. 중첩 Sequential의 Linear도 교체되어야 합니다.',node='replace_linear_with_lora',post='replace_linear_with_lora(model,rank,alpha)',kind='replace')
add('6L',3,'기존 가중치 동결 후 LoRA 주입','model, LORA_RANK, LORA_ALPHA',"model=nn.Sequential(nn.Linear(3,4),nn.Sequential(nn.ReLU(),nn.Linear(4,2))); LORA_RANK=case+1; LORA_ALPHA=2.",'model','원본 파라미터 requires_grad=False → 새 LoRA 파라미터 생성. 새 A/B만 학습 가능해야 합니다. 순서를 바꾸면 LoRA까지 동결될 수 있습니다.',span=(63,70),kind='freeze')

OBSERVE='''def observe(actual, kind, case, context):
    if kind=='optimizer':
        model=context['model']
        assert {id(p) for g in actual.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
        assert isinstance(actual,torch.optim.AdamW)
        return actual.param_groups[0]['lr'],actual.param_groups[0]['weight_decay']
    if kind=='loss':
        assert actual.ndim==0
        actual.backward()
        gradients=tuple(p.grad for p in context['model'].parameters())
        assert any(g is not None and g.abs().sum()>0 for g in gradients)
        return snapshot((actual,gradients))
    if kind=='classification':
        m,fs=actual
        assert m.out_head.out_features==context['num_classes']
        assert all(not p.requires_grad for p in m.emb.parameters())
        assert all(not p.requires_grad for p in m.trf_blocks[0].parameters())
        assert all(p.requires_grad for p in m.trf_blocks[-1].parameters())
        assert all(p.requires_grad for p in m.final_norm.parameters())
        assert all(p.requires_grad for p in m.out_head.parameters())
        return snapshot((m(torch.tensor([[1,2],[3,4]])), fs, tuple(m.out_head.weight.shape)))
    if kind in ('attention','norm','ff','block','gpt','lora','wrapped','replace','freeze'):
        m=actual
        m.eval()
        if kind=='attention':
            x=torch.randn(case+1,4,context['d_in'],requires_grad=True)
        elif kind=='norm':
            with torch.no_grad():
                m.scale.copy_(torch.arange(context['emb_dim'])+.5)
                m.shift.copy_(torch.arange(context['emb_dim'])-.7)
            x=torch.randn(2,3,context['emb_dim'],requires_grad=True)
        elif kind in ('ff','block'):x=torch.randn(2,3,context['cfg']['emb_dim'],requires_grad=True)
        elif kind=='gpt':x=torch.tensor([[1,2,3],[3,2,1]])
        else:
            dim=context['in_dim'] if kind=='lora' else 3
            x=torch.randn(2,3,dim,requires_grad=True)
            with torch.no_grad():
                for name,p in m.named_parameters():
                    if name=='B' or name.endswith('.B'):p.copy_(torch.arange(p.numel()).reshape_as(p)*.05+.1)
        y=m(x)
        if kind=='attention':
            z=x.detach().clone();z[:,-1]+=10
            torch.testing.assert_close(m(z)[:,:-1],y[:,:-1],rtol=1e-5,atol=1e-6)
            assert 'mask' in dict(m.named_buffers())
            q=m.W_query(x);k=m.W_key(x);v=m.W_value(x)
            expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
            torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
        if kind=='norm':
            expected=torch.nn.functional.layer_norm(x,(context['emb_dim'],),m.scale,m.shift,m.eps)
            torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
        y.square().mean().backward()
        grads=tuple((name,None if p.grad is None else p.grad.detach().clone()) for name,p in m.named_parameters())
        if kind=='freeze':
            for name,p in m.named_parameters():
                assert p.requires_grad == (name.endswith('.A') or name.endswith('.B')),name
        return snapshot((y,flags(m),grads))
    return snapshot(actual)
'''
SETUP+='\n\n'+OBSERVE

def literal(x):
 if isinstance(x,torch.Tensor):return f'torch.tensor({x.tolist()!r},dtype={x.dtype})'
 if isinstance(x,dict):return '{'+', '.join(repr(k)+':'+literal(v) for k,v in x.items())+'}'
 if isinstance(x,tuple):return '('+', '.join(literal(v) for v in x)+',)'
 return repr(x)

def build():
 env={};exec(SETUP,env)
 cells=[nb.v4.new_markdown_cell('# LLM 빈칸 실습 모음 · Chapter 2~6\n\n**문제 → 빈칸 코드 → 검증 코드 → 접힌 해설·정답** 순서입니다. 준비 셀을 한 번 실행한 뒤 원하는 문제부터 풀 수 있습니다. CPU PyTorch만 필요하며 다운로드·실제 LLM·GPT-2 토크나이저 설치는 없습니다.\n\n원본 `???`만 `MISSING`으로 바꿔 문법적으로 실행 가능한 빈칸을 만들었습니다. **MISSING을 알맞은 코드로 교체**하세요. 따옴표 안 MISSING은 문자열 키, `.MISSING`은 메서드/필드, 단독 MISSING은 식/인자입니다. 원본 함수·클래스를 작은 solve 함수 안에 넣었으며 나머지 문맥은 제공합니다.\n\nSyntaxError가 난 경우 이전 함수가 남을 수 있습니다. 수정 후 문제 셀부터 다시 실행하세요. PASS는 아래 더미 입력의 검사 결과이며 실제 GPT 품질이나 학습자 숙련을 보장하지 않습니다.\n\nChapter2 Tokenizer 비교 이후·다운로드는 제외했습니다. 해설은 먼저 시도한 뒤 펼치세요. 준비 셀의 보조 구현은 문제 간 의존성을 없애기 위한 제공 코드입니다. 출제 공식 답안 파일은 지정되지 않아 원본 힌트와 수식으로 참고 답안을 구성했습니다.'),nb.v4.new_code_cell(SETUP)]
 covered=set(); report=[]; mapping=[]
 for ix,s in enumerate(spec,1):
  name=f'solve_{ix:02d}'
  def wrap(body):
   return f'def {name}({s["args"]}):\n'+textwrap.indent('\n'.join(p for p in [s['pre'],body,s['post'],'return '+s['ret']] if p),'    ')
  solution=wrap(s['answer'])
  masked=re.sub(r'\?{3,}','MISSING',s['raw'])
  masked='\n'.join(l+'  # ???? 원본 빈칸' if 'MISSING' in l else l for l in masked.splitlines())
  blank='import torch\nimport torch.nn as nn\n\n'+wrap(masked)
  compile(blank,'student','exec');exec(solution,env)
  for line,col in LOC[(s['f'],s['c'])]:
   if s['bounds'][0]<=line<s['bounds'][1]:
    key=(s['f'],s['c'],line,col);assert key not in covered,('duplicate',key);covered.add(key)
  expected=[]
  for case in range(2):
   torch.manual_seed(2000+ix*10+case);env['case']=case;exec(s['fixture'],env)
   actual=eval(f'{name}({s["args"]})',env)
   expected.append(env['observe'](actual,s['kind'],case,env))
   if s['check']:exec(s['check'],env)
  test=f'''# 정답/실패 확인. 이 셀의 입력과 기대값은 수정하지 마세요.
def check_{ix:02d}():
    expected={literal(tuple(expected))}
    for case in range(2):
        torch.manual_seed({2000+ix*10}+case)
'''+textwrap.indent(s['fixture'].replace('; ','\n'),'        ')+f'\n        actual={name}({s["args"]})\n        observed=observe(actual,{s["kind"]!r},case,locals())\n        same(observed,expected[case])\n'
  if s['check']:test+=textwrap.indent(s['check'],'        ')+'\n'
  test+=f'''    print('PASS · {ix:02d} · 두 입력 및 shape/dtype/동작 검증')
try:
    check_{ix:02d}()
except Exception as error:
    print('FAIL · {ix:02d} ·',type(error).__name__,str(error))
'''
  exec(test,env)
  # Fail on a genuinely incomplete definition; not just syntactically invalid cells.
  exec(blank,env)
  try:env[f'check_{ix:02d}']()
  except Exception:pass
  else:raise AssertionError(('blank accepted',ix))
  prompt=f'## {ix:02d}. {s["title"]}\n\n원본: `{FILES[s["f"]]}` **전체 셀 {s["c"]+1}** (Markdown 포함, 1부터), 원본 줄 {s["bounds"][0]+1}~{s["bounds"][1]}.\n\n**목표:** 원본 빈칸을 완성하세요. 반환값: `{s["ret"]}`.\n\n**제공 입력/인자 예시:**\n```python\ncase=0\n{s["fixture"].replace("; ",chr(10))}\n```\n\n테스트는 서로 다른 입력 두 개를 사용합니다. 작은 CPU 텐서로 shape·값·필요 시 gradient를 검사합니다.'
  explanation=f'<details><summary>해설 · 쉬운 공식 · 참고 답안 (시도 후 펼치세요)</summary>\n\n{s["explain"]}\n\n```python\n{solution}\n```\n\n다운로드와 큰 실행 문맥은 제외했습니다. 참고 답안은 원본의 TODO 힌트에 근거합니다.\n</details>'
  cells.extend([nb.v4.new_markdown_cell(prompt),nb.v4.new_code_cell(blank),nb.v4.new_code_cell(test),nb.v4.new_markdown_cell(explanation)])
  mapping.append(f'| {ix:02d} | {s["f"]} | {s["c"]+1} | {s["title"]} |')
  report.append({'question':ix,'file':FILES[s['f']],'cell_1based':s['c']+1,'cases':2,'reference_pass':True,'blank_rejected':True})
 allholes={(f,c,line,col) for (f,c),loc in LOC.items() for line,col in loc}
 assert covered==allholes,('uncovered',allholes-covered)
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 nb.validate(notebook);nb.write(notebook,OUT/'llm-exercises.ipynb')
 (OUT/'validation.json').write_text(json.dumps({'questions':len(spec),'source_blank_count':len(covered),'checks':report},indent=2),encoding='utf-8')
 (OUT/'README.md').write_text('# LLM 경량 빈칸 실습\n\n[통합 노트북](llm-exercises.ipynb)을 열어 준비 셀 → 문제 코드 → 검증 코드 순서로 실행하세요. `MISSING`은 원본 `???`입니다. 정답으로 교체하고 함수 셀부터 재실행하세요. 해설·답은 각 문제 아래 접혀 있습니다.\n\n- 지정한 6개 파일의 코드상 빈칸 '+str(len(covered))+'개를 '+str(len(spec))+'문제로 묶었습니다. 주석 속 ???는 세지 않습니다.\n- CPU PyTorch만 필요합니다. 다운로드·모델 체크포인트·tiktoken·pandas 없이 실행합니다.\n- 더미 토크나이저는 UTF-8 바이트 인코딩입니다. 실제 BPE 분할 품질·ID는 평가하지 않고 encode/decode 호출과 텐서 흐름만 검사합니다.\n- 분류의 pandas iloc는 작은 더미로 대체했습니다. Chapter2 Tokenizer 비교 이후는 기존 시험 범위에 따라 제외했습니다.\n- Attention과 GPT는 원본 소형 구조를 사용합니다. 생성은 다음 ID를 예측하는 더미 모델입니다.\n- Chapter5 원본 generate는 주석과 달리 top-k/temperature 구현이 없으며 greedy입니다. EOS 테스트는 원본 조건문에 맞춰 배치1, EOS를 붙이기 전 종료합니다.\n- 학습은 두 배치만 실행하여 zero_grad와 업데이트를 검증합니다. LoRA는 B를 비영 값으로 바꿔 수식·gradient·동결 범위를 검사합니다.\n- 원본 클래스/함수의 중복 구현은 제공 보조 코드로 두어 문제끼리 독립적입니다. 실제 다운로드·전체 사전학습·평가 품질은 검사하지 않습니다.\n- 참고 답안은 별도 공식 답안 파일이 아니라 원본 힌트·수식으로 작성했습니다. 원본 파일은 변경하지 않았습니다.\n- 재생성: `python build_exercises.py` (nbformat 필요). 학습 답안이 덮어써지므로 작성 중 재생성하지 마세요.\n\n| 문제 | Chapter | 원본 셀(1부터) | 주제 |\n|---|---|---|---|\n'+'\n'.join(mapping)+'\n',encoding='utf-8')
 print('GENERATED',len(spec),'questions;',len(covered),'original blanks')

if __name__=='__main__':build()
