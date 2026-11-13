"""Generate small offline Vision exercises from the explicitly scoped course notebooks.
Rebuilding overwrites the student notebook; never run over a completed submission.
"""
import ast
import contextlib
import hashlib
import io
import json
import re
import textwrap
from pathlib import Path
import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = {'R': '01_ResNet18_CIFAR10.ipynb', 'V': '02_ViT_CIFAR10.ipynb',
         'D': '03_DETR.ipynb', 'U': '04_Unet.ipynb'}
LIMITS = {'R': 15, 'V': 23, 'D': 30, 'U': 15}
SETUP = '''import copy
import math
import numpy as np
import torch
import torch.nn as nn  # nn = torch.nn
import torch.nn.functional as F  # F = torch.nn.functional
import torchvision
import torchvision.transforms as T  # T = torchvision.transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

torch.set_num_threads(1)
MISSING = object()  # ???? 대신 실행 가능한 표식. 이 값을 올바른 코드로 바꾸세요.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

class ModeProbe(nn.Module):
    # train/eval과 gradient 모드를 기록하는 작은 제공 분류기입니다.
    def __init__(self, width=4, classes=3):
        super().__init__()
        self.fc = nn.Linear(width, classes)
        self.seen = []
    def forward(self, x):
        self.seen.append((self.training, torch.is_grad_enabled()))
        return self.fc(x)
'''
specs = []

def add(key, title, cells, criteria, task, code, check, explain, mutation):
    specs.append(dict(key=key, title=title, cells=cells, criteria=criteria,
                      task=task, code=textwrap.dedent(code).strip(),
                      check=textwrap.dedent(check).strip(), explain=explain, mutation=mutation))

# <<expression>> marks one expression; whole missing statements use a marker-only line.
add('R01', '학습·평가 전처리 구분', [6], '데이터 구성·이미지 변환',
    'size×size RandomCrop(padding=4), 좌우 반전, ToTensor, Normalize 순으로 train 변환을 구성하세요. test에는 ToTensor와 Normalize만 사용합니다. RGB PIL → float32 (3,size,size). 전달받은 mean/std를 사용하세요.',
    '''def solve(size, mean, std):
    train = T.Compose([
        <<T.RandomCrop(size, padding=4)>>,
        <<T.RandomHorizontalFlip()>>,
        <<T.ToTensor()>>,
        <<T.Normalize(mean, std)>>,
    ])
    test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train, test''',
    '''def check(fn):
    for size in (8, 12):
        mean, std = (0.2,0.3,0.4), (0.5,0.2,0.1)
        train, test = fn(size, mean, std)
        assert [type(t) for t in train.transforms] == [T.RandomCrop,T.RandomHorizontalFlip,T.ToTensor,T.Normalize]
        assert [type(t) for t in test.transforms] == [T.ToTensor,T.Normalize]
        assert train.transforms[0].size == (size,size) and train.transforms[0].padding == 4
        assert train.transforms[1].p == 0.5
        image = Image.fromarray(np.arange(size*size*3,dtype=np.uint8).reshape(size,size,3))
        expected=T.Normalize(mean,std)(T.ToTensor()(image))
        close(test(image),expected)
        torch.manual_seed(51)
        actual=train(image)
        torch.manual_seed(51)
        ref=T.Compose([T.RandomCrop(size,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean,std)])(image)
        close(actual,ref)
        assert actual.shape==(3,size,size) and actual.dtype==torch.float32''',
    'Compose는 변환을 순서대로 실행합니다. 예: uint8 픽셀255는 ToTensor 뒤1.0, mean=.5/std=.5이면 Normalize 뒤1.0입니다. 평가에는 무작위 증강을 넣지 않습니다. T의 원래 이름은 torchvision.transforms입니다.',
    ('T.RandomHorizontalFlip()', 'T.RandomHorizontalFlip(p=0.0)'))

add('R02', 'DataLoader로 이미지·정답 묶기', [6], '데이터 구성',
    '제공된 train_set/test_set으로 로더를 만드세요. batch_size는 인자를 사용하고 train만 shuffle=True입니다. CPU 테스트를 위해 num_workers=0. 마지막 작은 배치는 버리지 않습니다.',
    '''def solve(train_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=<<True>>, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=<<False>>, num_workers=0)
    return train_loader, test_loader''',
    '''def check(fn):
    for bs in (2,3):
        ds=TensorDataset(torch.arange(5*3*4*4).float().reshape(5,3,4,4),torch.arange(5))
        tr,te=fn(ds,ds,bs)
        assert isinstance(tr.sampler,torch.utils.data.RandomSampler)
        assert isinstance(te.sampler,torch.utils.data.SequentialSampler)
        assert tr.num_workers==te.num_workers==0 and tr.batch_size==te.batch_size==bs
        batches=list(te)
        assert len(batches)==math.ceil(5/bs)
        close(torch.cat([y for x,y in batches]),torch.arange(5))
        assert batches[0][0].shape==(bs,3,4,4) and batches[0][1].dtype==torch.int64''',
    '예: 이미지5장, batch_size=2이면 배치 크기는2,2,1입니다. images=(B,3,H,W), labels=(B,) 정수 클래스 번호입니다. 원본의 worker/pin_memory는 실행환경 설정이라 여기서는 CPU용으로 고정했습니다.',
    ('shuffle=True','shuffle=False'))

add('R03', '역정규화와 broadcasting', [7], '데이터 구성·이미지 변환',
    '정규화된 배치 x=(B,3,H,W)를 x*std+mean으로 되돌리고 표시용으로 [0,1]에 제한하세요. mean/std를 (1,3,1,1)로 만들어 채널별로 적용합니다.',
    '''def solve(x, mean_values, std_values):
    mean = torch.tensor(mean_values, device=x.device).view(<<1,3,1,1>>)
    std = torch.tensor(std_values, device=x.device).view(<<1,3,1,1>>)
    return <<(x * std + mean).clamp(0,1)>>''',
    '''def check(fn):
    for b,h,w in [(1,4,5),(2,3,7)]:
        x=torch.linspace(-4,4,b*3*h*w).reshape(b,3,h,w)
        mean=(0.1,0.4,0.8);std=(0.2,0.3,0.5)
        expected=torch.stack([(x[:,c]*std[c]+mean[c]).clamp(0,1) for c in range(3)],dim=1)
        close(fn(x,mean,std),expected)''',
    '예: R채널 정규화 값2, std=.2, mean=.5 → .9. mean/std의1은 배치·높이·너비 방향으로 반복 적용됩니다. clamp는 정규화를 취소하는 연산이 아니라 표시 범위를 자르는 연산입니다.',
    ('x * std + mean','x / std + mean'))

add('R04', 'CIFAR용 ResNet stem과 분류기', [9], '레이어 구성·동작',
    '제공된 ResNet18에서 conv1을 3→64, kernel=3,stride=1,padding=1,bias=False로 바꾸세요. maxpool을 Identity로 대체하고 FC 출력은 num_classes로 바꿉니다. 모델 생성·다운로드는 제공 코드가 처리합니다.',
    '''def solve(model, num_classes):
    model.conv1 = <<nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)>>
    model.maxpool = <<nn.Identity()>>
    model.fc = <<nn.Linear(model.fc.in_features,num_classes)>>
    return model''',
    '''def check(fn):
    for classes in (3,10):
        m=fn(torchvision.models.resnet18(weights=None),classes).eval()
        assert m.conv1.in_channels==3 and m.conv1.out_channels==64
        assert m.conv1.kernel_size==(3,3) and m.conv1.stride==(1,1) and m.conv1.padding==(1,1)
        assert m.conv1.bias is None and isinstance(m.maxpool,nn.Identity)
        assert m.fc.in_features==512 and m.fc.out_features==classes
        with torch.no_grad():
            x=torch.randn(2,3,32,32)
            assert m.conv1(x).shape==(2,64,32,32)
            assert m(x).shape==(2,classes)''',
    '예: (2,3,32,32)→stem(2,64,32,32)→ResNet stages→pool(2,512,1,1)→flatten(2,512)→FC(2,10)。Identity는 입력을 그대로 통과시킵니다. 내부 residual 구조는 torchvision이 제공합니다.',
    ('stride=1','stride=2'))

add('R05', '분류 loss와 학습할 파라미터 연결', [13], '예측·정답을 학습으로 연결',
    'CrossEntropyLoss(label_smoothing 인자)와 SGD(lr,momentum,weight_decay 인자)를 생성하세요. optimizer는 model.parameters()를 갱신해야 합니다. logits=(B,C), labels=(B,) int64.',
    '''def solve(model, lr, momentum, weight_decay, smoothing):
    criterion = <<nn.CrossEntropyLoss(label_smoothing=smoothing)>>
    optimizer = <<torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)>>
    return criterion, optimizer''',
    '''def check(fn):
    for smoothing in (0.,0.1):
        model=nn.Linear(4,3)
        criterion,opt=fn(model,.03,.8,.002,smoothing)
        assert isinstance(criterion,nn.CrossEntropyLoss) and isinstance(opt,torch.optim.SGD)
        g=opt.param_groups[0]
        assert (g['lr'],g['momentum'],g['weight_decay'])==(.03,.8,.002)
        assert {id(p) for p in g['params']}=={id(p) for p in model.parameters()}
        logits=torch.tensor([[1.,-2.,.5],[.2,.3,2.]],requires_grad=True);labels=torch.tensor([0,2])
        close(criterion(logits,labels),F.cross_entropy(logits,labels,label_smoothing=smoothing))''',
    'CrossEntropyLoss에는 softmax 전 logits를 넣습니다. 예: labels=[2]는 세 번째 클래스가 정답이라는 뜻입니다. optimizer는 loss를 계산하지 않고 backward가 만든 gradient를 이용해 weight를 갱신합니다.',
    ('label_smoothing=smoothing','label_smoothing=0.3'))

add('R06', 'AMP 학습 한 배치', [11], '예측·정답을 학습으로 연결',
    'train 모드→gradient 초기화→forward/loss→scaled backward→optimizer step→scaler update를 완성하세요. images=(B,4), labels=(B,)인 작은 분류기와 비활성 GradScaler를 제공합니다. GPU autocast 문맥은 CPU 최소 검증에서 생략합니다.',
    '''def solve(model, images, labels, optimizer, criterion, scaler):
    model.train()
    <<optimizer.zero_grad(set_to_none=True)>>
    logits = model(images)
    loss = <<criterion(logits,labels)>>
    <<scaler.scale(loss).backward()>>
    <<scaler.step(optimizer)>>
    <<scaler.update()>>
    return loss.item()''',
    '''def check(fn):
    for seed in (12,21):
        torch.manual_seed(seed)
        a=ModeProbe(); b=copy.deepcopy(a); a.eval()
        oa=torch.optim.SGD(a.parameters(),lr=.1); ob=torch.optim.SGD(b.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        for size in (3,2):
            x=torch.randn(size,4);y=torch.arange(size)%3
            ob.zero_grad(set_to_none=True); expected=F.cross_entropy(b(x),y);expected.backward();ob.step()
            got=fn(a,x,y,oa,nn.CrossEntropyLoss(),scaler)
            assert abs(got-expected.item())<1e-6
            for p,q in zip(a.parameters(),b.parameters()):close(p,q)
        assert a.seen==[(True,True),(True,True)]''',
    '예: batch1의 gradient를 지우지 않고 batch2에서 backward하면 두 배치 gradient가 누적됩니다. zero_grad는 이를 초기화합니다. scaler.step이 optimizer를 호출해 weight를 갱신하고, scaler.update는 배율을 갱신합니다. CPU에서는 배율 처리를 비활성화하고 같은 순서를 검증합니다.',
    ('optimizer.zero_grad(set_to_none=True)','pass'))

add('R07', '평가 모드·정확도·샘플 가중 평균', [11], '평가 Metric·학습/평가 구분',
    '빈칸을 채워 평가 loss와 top-1 accuracy를 반환하세요. 각 배치 criterion은 평균 loss입니다. 마지막 배치 크기가 달라도 전체 샘플 평균이 맞아야 합니다. eval과 no_grad 모두 필요합니다.',
    '''def solve(model, loader, criterion):
    <<model.eval()>>
    total_loss, correct, n = 0., 0, 0
    with <<torch.no_grad()>>:
        for images, labels in loader:
            logits=model(images)
            bs=images.size(0)
            total_loss += <<criterion(logits,labels).item() * bs>>
            correct += <<(logits.argmax(dim=1)==labels).sum().item()>>
            n += bs
    return total_loss/n, correct/n''',
    '''def check(fn):
    for bs in (2,3):
        torch.manual_seed(9);m=ModeProbe();m.train()
        x=torch.randn(5,4);y=torch.tensor([0,1,2,0,1])
        logits=F.linear(x,m.fc.weight,m.fc.bias).detach()
        expected_loss=F.cross_entropy(logits,y).item()
        expected_acc=(logits.argmax(1)==y).float().mean().item()
        got=fn(m,DataLoader(TensorDataset(x,y),batch_size=bs),nn.CrossEntropyLoss())
        assert abs(got[0]-expected_loss)<1e-6 and abs(got[1]-expected_acc)<1e-6
        assert all(state==(False,False) for state in m.seen)
        assert all(p.grad is None for p in m.parameters())''',
    '예: 2장 배치 loss1, 1장 배치 loss4 → 전체 평균=(1×2+4×1)/3=2입니다. (1+4)/2=2.5는 틀립니다. accuracy는 맞힌 샘플 수/전체 수이고 confidence 평균이 아닙니다.',
    ('criterion(logits,labels).item() * bs','criterion(logits,labels).item()'))

add('R08', '추론 확률·클래스·confidence', [15], '레이어 활용·평가 Metric',
    'logits=(B,C)를 클래스 축 softmax로 확률로 바꾸고, 각 이미지의 최대 확률 conf=(B,), 클래스 pred=(B,)를 반환하세요. 반환 순서는 pred,conf,probs입니다.',
    '''def solve(logits):
    probs = <<F.softmax(logits,dim=1)>>
    conf, pred = <<probs.max(dim=1)>>
    return pred,conf,probs''',
    '''def check(fn):
    for x in [torch.tensor([[0.,0.,0.],[0.,math.log(2),0.]]),torch.tensor([[4.,1.],[1.,3.],[0.,-1.]])]:
        pred,conf,probs=fn(x)
        exp=x.exp()/x.exp().sum(1,keepdim=True)
        close(probs,exp);close(probs.sum(1),torch.ones(x.shape[0]))
        close(pred,x.argmax(1));close(conf,exp.gather(1,pred[:,None]).squeeze(1))
        assert pred.dtype==torch.int64''',
    '예: logits=[0,ln2,0]→확률=[.25,.5,.25]→pred=1,conf=.5. 정답 labels와 비교해야 accuracy를 알 수 있습니다. confidence=.5만으로 맞았다고 판단하지 않습니다.',
    ('dim=1','dim=0'))

add('V01', '이미지 패치화와 선형 임베딩', [11], '레이어 구성·데이터 구성',
    '이미지의 패치를 (p1,p2,c) 순서로 펼친 후 Linear(patch_dim,dim)를 적용하는 Sequential을 만드세요. 입력 (B,C,H,W), 패치(ph,pw), 출력 (B,N,dim). H,W는 패치 크기로 나누어집니다.',
    '''def solve(channels, ph, pw, dim):
    patch_dim = <<channels * ph * pw>>
    return nn.Sequential(
        Rearrange(<<"b c (h p1) (w p2) -> b (h w) (p1 p2 c)">>,p1=ph,p2=pw),
        <<nn.Linear(patch_dim,dim)>>,
    )''',
    '''def check(fn):
    for c,ph,pw,h,w,d in [(3,2,2,4,4,2),(2,2,3,4,6,5)]:
        layer=fn(c,ph,pw,d)
        x=torch.arange(2*c*h*w).float().reshape(2,c,h,w)
        patches=[]
        for y in range(0,h,ph):
            for z in range(0,w,pw):patches.append(x[:,:,y:y+ph,z:z+pw].permute(0,2,3,1).reshape(2,-1))
        expected=torch.stack(patches,dim=1)
        close(layer[0](x),expected)
        assert layer[1].in_features==c*ph*pw and layer[1].out_features==d
        close(layer(x),F.linear(expected,layer[1].weight,layer[1].bias))''',
    '예: (B,3,4,4),패치2×2 → 패치4개,패치당12개 숫자 → (B,4,12)→Linear(12,2)→(B,4,2). Rearrange는 값의 순서·shape만 바꾸고 Linear가 학습 가능한 투영을 합니다.',
    ('(p1 p2 c)','(c p1 p2)'))

add('V02', 'CLS 토큰과 위치 임베딩', [11], '레이어 구성·동작',
    'x=(B,N,D), cls_token=(1,1,D), pos_embedding=(1,M,D), M>=N+1입니다. CLS를 배치만큼 복제하여 토큰 축 맨 앞에 붙이고 필요한 위치 임베딩만 더하세요.',
    '''def solve(x, cls_token, pos_embedding):
    b,n,_=x.shape
    cls_tokens=<<repeat(cls_token,"1 1 d -> b 1 d",b=b)>>
    x=<<torch.cat((cls_tokens,x),dim=1)>>
    return <<x + pos_embedding[:,:(n+1)]>>''',
    '''def check(fn):
    for b,n,d in [(1,4,2),(3,6,4)]:
        x=torch.arange(b*n*d).float().reshape(b,n,d)
        cls=torch.full((1,1,d),-2.,requires_grad=True)
        pos=torch.arange((n+3)*d).float().reshape(1,n+3,d).requires_grad_()
        got=fn(x,cls,pos)
        expected=torch.stack([torch.vstack([cls[0],x[i]])+pos[0,:n+1] for i in range(b)])
        close(got,expected);assert got.shape==(b,n+1,d)
        got.sum().backward();close(cls.grad,torch.full_like(cls,float(b)))''',
    '예: 패치4개×차원2 → CLS 추가 후5개×차원2. 위치0은 CLS, 위치1~4는 패치입니다. 위치 임베딩은 새 토큰으로 cat하는 것이 아니라 각 위치 벡터에 더합니다.',
    ('pos_embedding[:,:(n+1)]','pos_embedding[:,1:(n+2)]'))

add('V03', 'PreNorm: 먼저 LayerNorm', [9], '레이어 구성·동작',
    'PreNorm의 forward 일부입니다. 제공된 norm은 LayerNorm(D), fn은 D→D 모듈입니다. 마지막 임베딩 축을 정규화한 뒤 fn에 전달하세요.',
    '''def solve(x, norm, fn):
    return <<fn(norm(x))>>''',
    '''def check(fn):
    for d in (2,5):
        torch.manual_seed(d);x=torch.randn(2,3,d);norm=nn.LayerNorm(d);layer=nn.Linear(d,d)
        got=fn(x,norm,layer)
        z=(x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,keepdim=True,unbiased=False)+norm.eps)
        close(got,F.linear(z*norm.weight+norm.bias,layer.weight,layer.bias))''',
    '예: 각 토큰 [1,3]을 먼저 평균0 근처로 정규화하고 Attention/FFN에 넣습니다. shape (B,N,D)는 유지됩니다. fn(norm(x))와 norm(fn(x))는 순서가 달라 결과도 다릅니다.',
    ('fn(norm(x))','norm(fn(x))'))

add('V04', 'FFN의 확장·활성화·축소', [9], '레이어 구성·동작',
    'Linear(dim,hidden_dim)→ReLU→Dropout→Linear(hidden_dim,dim)→Dropout의 FFN을 만드세요. 주의: 지정 원본의 주석은 GELU지만 실제 코드는 ReLU이므로 이 문제는 ReLU를 요구합니다.',
    '''def solve(dim, hidden_dim, dropout):
    return nn.Sequential(
        <<nn.Linear(dim,hidden_dim)>>,
        <<nn.ReLU()>>,
        nn.Dropout(dropout),
        <<nn.Linear(hidden_dim,dim)>>,
        nn.Dropout(dropout),
    )''',
    '''def check(fn):
    for d,hidden,p in [(2,5,.2),(4,7,.4)]:
        m=fn(d,hidden,p).eval()
        assert [type(a) for a in m]==[nn.Linear,nn.ReLU,nn.Dropout,nn.Linear,nn.Dropout]
        assert m[2].p==m[4].p==p
        x=torch.randn(2,3,d)
        assert m[0](x).shape==(2,3,hidden)
        expected=F.linear(F.relu(F.linear(x,m[0].weight,m[0].bias)),m[3].weight,m[3].bias)
        close(m(x),expected);assert m(x).shape==x.shape''',
    '예: (2,5,4)→Linear(4,8)→(2,5,8)→ReLU→Linear(8,4)→(2,5,4). 토큰 개수5는 유지됩니다. 여기에는 LayerNorm을 중복 삽입하지 않고 바깥 PreNorm이 담당합니다.',
    ('nn.ReLU()','nn.GELU()'))

add('V05', 'QKV 분리와 head 축 만들기', [9], '레이어 구성·동작',
    'to_qkv는 D→3*heads*dim_head인 Linear입니다. x=(B,N,D)를 투영하고 마지막 축을 3등분한 뒤 각각 (B,heads,N,dim_head)로 바꾸세요. Q,K,V 순서로 반환합니다.',
    '''def solve(x, to_qkv, heads):
    qkv=<<to_qkv(x).chunk(3,dim=-1)>>
    q,k,v=map(lambda t: <<rearrange(t,"b n (h d) -> b h n d",h=heads)>>,qkv)
    return q,k,v''',
    '''def check(fn):
    for b,n,d,h,dh in [(2,5,4,2,3),(1,4,3,3,2)]:
        x=torch.randn(b,n,d);layer=nn.Linear(d,3*h*dh,bias=False)
        got=fn(x,layer,h); raw=F.linear(x,layer.weight)
        for i,t in enumerate(got):
            expected=raw[:,:,i*h*dh:(i+1)*h*dh].reshape(b,n,h,dh).permute(0,2,1,3)
            close(t,expected);assert t.shape==(b,h,n,dh)''',
    '예: B2,N5,D4,heads2,dim_head3이면 Linear 출력(2,5,18)→Q,K,V 각(2,5,6)→각(2,2,5,3). 입력D와 heads×dim_head가 꼭 같아야 하는 구현이 아닙니다.',
    ('chunk(3,dim=-1)','chunk(3,dim=1)'))

add('V06', 'Scaled attention과 V 가중합', [9], '레이어 구성·동작',
    'q,k,v=(B,H,N,d)입니다. QKᵀ에 d**-0.5를 곱하고 key축 softmax를 적용한 뒤 V를 가중합하세요. 반환 attn=(B,H,N,N),out=(B,H,N,d). ViT에는 causal mask를 넣지 않습니다.',
    '''def solve(q,k,v):
    scale=<<q.shape[-1] ** -0.5>>
    dots=<<torch.einsum("b h i d,b h j d->b h i j",q,k) * scale>>
    attn=<<dots.softmax(dim=-1)>>
    out=<<torch.einsum("b h i j,b h j d->b h i d",attn,v)>>
    return attn,out''',
    '''def check(fn):
    for b,h,n,d in [(1,2,3,2),(2,3,4,3)]:
        torch.manual_seed(n);q=torch.randn(b,h,n,d);k=torch.randn_like(q);v=torch.randn_like(q)
        attn,out=fn(q,k,v)
        scores=(q@k.transpose(-2,-1))/math.sqrt(d)
        expected=(scores-scores.max(-1,keepdim=True).values).exp()
        expected=expected/expected.sum(-1,keepdim=True)
        close(attn,expected);close(out,expected@v)
        close(attn.sum(-1),torch.ones(b,h,n));assert (attn>0).all()''',
    '예: 한 query의 weight=[.2,.8], V=[[1,0],[0,3]]이면 새 벡터=[.2,2.4]. i는 query, j는 key, d는 내적하는 특징축입니다. softmax 전 값은 score이고 후가 가중치입니다.',
    ('dots.softmax(dim=-1)','dots.softmax(dim=-2)'))

add('V07', 'head 합치기와 출력 투영', [9], '레이어 구성·동작',
    'out=(B,H,N,d)를 (B,N,H*d)로 재배열하고 제공된 to_out으로 투영하세요. heads를 섞지 않게 n과 h 축 순서를 주의하세요.',
    '''def solve(out,to_out):
    out=<<rearrange(out,"b h n d -> b n (h d)")>>
    return <<to_out(out)>>''',
    '''def check(fn):
    for b,h,n,d,od in [(2,2,3,2,3),(1,3,4,2,5)]:
        x=torch.arange(b*h*n*d).float().reshape(b,h,n,d)
        layer=nn.Linear(h*d,od)
        expected=F.linear(x.permute(0,2,1,3).reshape(b,n,h*d),layer.weight,layer.bias)
        close(fn(x,layer),expected)''',
    '예: (2,2,5,3)→(2,5,6)→Linear(6,4)→(2,5,4). 단순 reshape만 하면 토큰/헤드의 숫자 순서가 잘못 섞일 수 있습니다.',
    ('rearrange(out,"b h n d -> b n (h d)")','out.reshape(out.shape[0],out.shape[2],-1)'))

add('V08', '두 residual 연결의 순서', [9], '레이어 구성·동작',
    'attn과 ff에는 PreNorm이 이미 포함돼 있습니다. attention 결과에 입력을 더하고, 갱신된 x로 FFN을 계산해 다시 잔차를 더하세요. shape는 (B,N,D)로 유지됩니다.',
    '''def solve(x,attn,ff):
    x=<<attn(x)+x>>
    x=<<ff(x)+x>>
    return x''',
    '''def check(fn):
    for d in (2,4):
        x=torch.linspace(-2,2,2*3*d).reshape(2,3,d)
        a=lambda t:t*2+1
        f=lambda t:t.square()
        mid=3*x+1
        close(fn(x,a,f),mid.square()+mid)''',
    '예: x=1,attn(x)=2x이면 첫 결과3. ff(t)=t²이면 두 번째 결과9+3=12. 두 번째 skip의 x는 최초1이 아니라 갱신된3입니다.',
    ('ff(x)+x','ff(x)'))

add('V09', 'CLS/mean pooling과 클래스 logits', [11], '레이어 구성·동작',
    'x=(B,N+1,D)에서 pool="cls"면0번 토큰, "mean"이면 토큰축 평균을 선택합니다. 제공 mlp_head(LayerNorm→Linear)에 넣어 (B,C) logits를 반환하세요. 원본 mean은 CLS도 포함합니다.',
    '''def solve(x,pool,mlp_head):
    x=<<x[:,0] if pool=="cls" else x.mean(dim=1)>>
    return <<mlp_head(x)>>''',
    '''def check(fn):
    for pool in ('cls','mean'):
        x=torch.randn(2,5,4);head=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,3))
        picked=x[:,0,:] if pool=='cls' else sum(x[:,i,:] for i in range(5))/5
        close(fn(x,pool,head),head(picked));assert fn(x,pool,head).shape==(2,3)''',
    '예: (2,5,4)에서 CLS 선택→(2,4)→head→(2,10). 마지막 softmax는 분류 loss 입력 전에 넣지 않습니다. 토큰마다 클래스10개를 출력하는 문제가 아닙니다.',
    ('x[:,0]','x[:,1]'))

add('V10', 'log_softmax와 NLLLoss 학습', [19], '예측·정답을 학습으로 연결',
    '원본 ViT의 한 배치 학습 코드를 완성하세요. logits에 log_softmax(dim=1)를 적용한 뒤 nll_loss를 사용합니다. 반환값은 float loss. 모델·optimizer·데이터는 제공됩니다.',
    '''def solve(model,optimizer,data,target):
    model.train()
    optimizer.zero_grad()
    output=<<F.log_softmax(model(data),dim=1)>>
    loss=<<F.nll_loss(output,target)>>
    <<loss.backward()>>
    <<optimizer.step()>>
    return loss.item()''',
    '''def check(fn):
    for seed in (3,6):
        torch.manual_seed(seed);a=ModeProbe();b=copy.deepcopy(a);a.eval()
        oa=torch.optim.SGD(a.parameters(),lr=.02);ob=torch.optim.SGD(b.parameters(),lr=.02)
        for size in (2,3):
            x=torch.randn(size,4);y=torch.arange(size)%3
            ob.zero_grad();loss=F.cross_entropy(b(x),y);loss.backward();ob.step()
            assert abs(fn(a,oa,x,y)-loss.item())<1e-6
            for p,q in zip(a.parameters(),b.parameters()):close(p,q)
        assert all(t==(True,True) for t in a.seen)''',
    '예: 정답 클래스 확률.8이면 NLL=-log(.8)。log_softmax+NLLLoss는 동일 logits에 CrossEntropyLoss를 적용한 것과 같습니다. softmax 확률을 nll_loss에 그대로 넣으면 다릅니다.',
    ('F.log_softmax','F.softmax'))

add('V11', '체크포인트에서 학습 상태 복원', [15,23], '학습에 필요한 작업·레이어 활용',
    '이미 메모리에 읽어 둔 ckpt를 제공합니다. model과 optimizer의 state_dict를 복원하고 epoch와 test_acc를 반환하세요. 파일 읽기·저장은 출제하지 않습니다.',
    '''def solve(model,optimizer,ckpt):
    <<model.load_state_dict(ckpt["model"])>>
    <<optimizer.load_state_dict(ckpt["optimizer"])>>
    return int(ckpt.get("epoch",0)),float(ckpt.get("test_acc",0.0))''',
    '''def check(fn):
    for epoch in (2,7):
        a=nn.Linear(2,3);opt=torch.optim.SGD(a.parameters(),lr=.07,momentum=.8)
        F.cross_entropy(a(torch.ones(2,2)),torch.tensor([0,1])).backward();opt.step()
        ckpt=copy.deepcopy(dict(model=a.state_dict(),optimizer=opt.state_dict(),epoch=epoch,test_acc=.6))
        b=nn.Linear(2,3);ob=torch.optim.SGD(b.parameters(),lr=.01,momentum=.0)
        assert fn(b,ob,ckpt)==(epoch,.6)
        for p,q in zip(a.parameters(),b.parameters()):close(p,q)
        assert ob.param_groups[0]['lr']==.07 and ob.param_groups[0]['momentum']==.8
        for p,q in zip(a.parameters(),b.parameters()):close(opt.state[p]['momentum_buffer'],ob.state[q]['momentum_buffer'])''',
    '예: epoch7,lr=.07,모멘텀 버퍼를 저장했다면 모델 weight뿐 아니라 optimizer 상태도 복원해야 그 상태에서 학습을 이어갑니다. 이 문제의 입력은 다운로드 없는 dict입니다.',
    ('optimizer.load_state_dict(ckpt["optimizer"])','pass'))

add('D01', 'DETR 입력 리사이즈·정규화·배치', [8,18], '데이터 구성·이미지 변환',
    'PIL RGB 이미지를 짧은 변 short_side에 맞춰 종횡비를 유지한 Resize→ToTensor→ImageNet Normalize 후 batch축을 추가하세요. 실제 원본은800이지만 테스트는8/10입니다.',
    '''def solve(image,short_side):
    transform=T.Compose([
        <<T.Resize(short_side)>>,
        T.ToTensor(),
        <<T.Normalize([.485,.456,.406],[.229,.224,.225])>>,
    ])
    return <<transform(image).unsqueeze(0)>>''',
    '''def check(fn):
    for side in (8,10):
        im=Image.fromarray(np.full((6,12,3),128,dtype=np.uint8))
        got=fn(im,side)
        assert got.shape==(1,3,side,2*side)
        for c,(mean,std) in enumerate(zip([.485,.456,.406],[.229,.224,.225])):
            close(got[0,c],torch.full((side,2*side),(128/255-mean)/std))''',
    '예: H6,W12에 Resize(8)→H8,W16。Resize((8,8))은 정사각형 강제 변환이라 다릅니다. 최종 입력은(1,3,8,16)입니다.',
    ('T.Resize(short_side)','T.Resize((short_side,short_side))'))

add('D02', '중심 박스를 픽셀 꼭짓점으로', [8], '데이터 변환·모델 출력 활용',
    'boxes=(N,4)는 정규화된 cx,cy,w,h입니다. xyxy로 변환하고 PIL size=(W,H)를 이용해 픽셀 좌표로 되돌리세요. 빈 박스 N=0도 처리해야 합니다.',
    '''def solve(boxes,size):
    cx,cy,w,h=boxes.unbind(1)
    corners=<<torch.stack([cx-.5*w,cy-.5*h,cx+.5*w,cy+.5*h],dim=1)>>
    img_w,img_h=size
    return <<corners * boxes.new_tensor([img_w,img_h,img_w,img_h])>>''',
    '''def check(fn):
    boxes=torch.tensor([[.5,.5,.2,.4],[.25,.25,.5,.5]])
    close(fn(boxes,(200,100)),torch.tensor([[80.,30.,120.,70.],[0.,0.,100.,50.]]))
    close(fn(boxes,(100,200)),torch.tensor([[40.,60.,60.,140.],[0.,0.,50.,100.]]))
    assert fn(torch.empty(0,4),(200,100)).shape==(0,4)''',
    '예: cx=.5,cy=.5,w=.2,h=.4→xyxy=[.4,.3,.6,.7]。이미지 W200,H100이면[80,30,120,70]입니다. 텐서는H,W 순서지만 PIL size는W,H 순서입니다.',
    ('[img_w,img_h,img_w,img_h]','[img_h,img_w,img_h,img_w]'))

add('D03', 'no-object 제외와 confidence 필터', [18], '모델 출력 활용·데이터 구성',
    'outputs의 pred_logits=(1,Q,C+1),pred_boxes=(1,Q,4)를 받습니다. 전체 C+1에 softmax 후 마지막 no-object 열을 제외하고, 최고 객체 확률이 threshold보다 큰 query만 남기세요. 반환은 선택된 객체확률,선택된 정규화박스,keep 마스크입니다.',
    '''def solve(outputs,threshold):
    probas=<<outputs['pred_logits'].softmax(-1)[0,:,:-1]>>
    keep=<<probas.max(-1).values > threshold>>
    return probas[keep],<<outputs['pred_boxes'][0,keep]>>,keep''',
    '''def check(fn):
    probs=torch.tensor([[[.95,.02,.03],[.03,.02,.95],[.4,.5,.1]]])
    boxes=torch.arange(12).float().reshape(1,3,4)/12
    outputs={'pred_logits':probs.log(),'pred_boxes':boxes}
    for threshold,ids in [(.9,[0]),(.45,[0,2]),(.99,[])]:
        p,b,keep=fn(outputs,threshold)
        expected=torch.tensor([i in ids for i in range(3)])
        close(keep,expected);close(p,probs[0,expected,:-1]);close(b,boxes[0,expected])
        assert keep.dtype==torch.bool''',
    '예: [고양이.03,개.02,no-object.95]인 query는 버립니다. no-object를 먼저 제거한 뒤 softmax하면 .6/.4로 재정규화되어 객체 확률을 부풀릴 수 있습니다. confidence threshold는 IoU threshold나 mAP가 아닙니다.',
    ("outputs['pred_logits'].softmax(-1)[0,:,:-1]","outputs['pred_logits'][0,:,:-1].softmax(-1)"))

add('D04', 'forward hook으로 attention 추출', [22], '레이어 활용',
    '제공 모듈은 forward에서 (결과,attention_weights)를 반환합니다. hook으로 output[1]을 목록에 저장하고 forward 1회 후 hook을 제거하세요. 반환 saved[0]은 가중치 텐서입니다.',
    '''def solve(module,x):
    saved=[]
    handle=module.register_forward_hook(<<lambda module,inputs,output: saved.append(output[1])>>)
    module(x)
    <<handle.remove()>>
    return saved[0]''',
    '''def check(fn):
    class ToyAttention(nn.Module):
        def forward(self,x):return x*2,x@x.transpose(-1,-2)
    for n in (3,5):
        m=ToyAttention();x=torch.randn(1,n,2)
        close(fn(m,x),x@x.transpose(-1,-2))
        assert len(m._forward_hooks)==0
        m(x);assert len(m._forward_hooks)==0''',
    '예: forward가 (output=(1,3,2),weights=(1,3,3))를 반환하면 output[1]만 저장합니다. hook의 output 인자는 이 반환 튜플 전체입니다. 완료 후 remove해야 다음 실행 때 중복 저장되지 않습니다.',
    ('saved.append(output[1])','saved.append(output[0])'))

add('D05', 'attention 토큰축을 공간으로 복원', [24,28,30], '레이어 동작·출력 활용',
    'encoder_attn=(1,HW,HW),decoder_attn=(1,Q,HW)를 제공합니다(헤드 평균 완료). encoder를(H,W,H,W)로 바꾸고, query 위치(qy,qx)가 보는 key지도(H,W)와 decoder query_id의 지도(H,W)를 반환하세요. 주의: 원본30셀의 마지막 두 축 슬라이스는 반대 방향입니다. 여기서는 query→key 의미에 맞게 앞 두 축을 고릅니다.',
    '''def solve(encoder_attn,decoder_attn,h,w,qy,qx,query_id):
    spatial=<<encoder_attn[0].reshape(h,w,h,w)>>
    enc_map=<<spatial[qy,qx,:,:]>>
    dec_map=<<decoder_attn[0,query_id].reshape(h,w)>>
    return enc_map,dec_map''',
    '''def check(fn):
    for h,w in [(2,3),(3,2)]:
        s=h*w;enc=torch.arange(s*s).float().reshape(1,s,s);dec=torch.arange(4*s).float().reshape(1,4,s)
        a,b=fn(enc,dec,h,w,1,0,2)
        close(a,enc[0,w,:].reshape(h,w));close(b,dec[0,2,:].reshape(h,w))''',
    '예: H2,W3에서 query(y1,x0)는1×3+0=3번 토큰입니다. encoder[0,3,:]가 이 query가 모든 key를 보는 행입니다. 원본 설명의 query/key 축 표기가 뒤바뀐 부분을 보정했습니다. 실제 head별 출력이면 먼저 head 선택/평균이 필요합니다.',
    ('spatial[qy,qx,:,:]','spatial[:,:,qy,qx]'))

add('U01', '이미지와 binary mask의 채널축', [5], '데이터 구성',
    '제공 NumPy float32 img/mask=(H,W)를 각각 (1,H,W) PyTorch 텐서로 바꾸세요. mask는 이미0/1이며 클래스 인덱스 long으로 바꾸지 않습니다. 반환 img,mask.',
    '''def solve(img,mask):
    img=<<np.expand_dims(img,axis=0)>>
    mask=<<np.expand_dims(mask,axis=0)>>
    return <<torch.from_numpy(img)>>,<<torch.from_numpy(mask)>>''',
    '''def check(fn):
    for h,w in [(4,5),(7,6)]:
        mask=(np.arange(h*w).reshape(h,w)%2).astype(np.float32);img=mask*.8
        a,b=fn(img,mask)
        close(a,torch.tensor(img)[None]);close(b,torch.tensor(mask)[None])
        assert a.dtype==b.dtype==torch.float32
        assert torch.stack([b,b]).shape==(2,1,h,w)''',
    '예: mask(4,4)→(1,4,4)→배치2장(2,1,4,4)。BCEWithLogitsLoss에서 prediction과 mask는 같은 shape의 float입니다. mask1은 전경,0은 배경입니다.',
    ('axis=0','axis=-1'))

add('U02', 'DoubleConv의 padding과 채널', [7], '레이어 구성·동작',
    'Conv3×3(padding1,bias=False)→ReLU(inplace=True)를 두 번 구성하세요. 채널은 in→mid→out입니다. 원본 실제 DoubleConv에는 BN이 없습니다.',
    '''def solve(in_channels,mid_channels,out_channels):
    return nn.Sequential(
        <<nn.Conv2d(in_channels,mid_channels,3,padding=1,bias=False)>>,
        nn.ReLU(inplace=True),
        <<nn.Conv2d(mid_channels,out_channels,3,padding=1,bias=False)>>,
        nn.ReLU(inplace=True),
    )''',
    '''def check(fn):
    for ci,cm,co in [(1,3,2),(2,4,3)]:
        m=fn(ci,cm,co)
        assert [type(t) for t in m]==[nn.Conv2d,nn.ReLU,nn.Conv2d,nn.ReLU]
        assert m[0].bias is None and m[2].bias is None
        assert m[1].inplace and m[3].inplace
        assert m[0].out_channels==cm and m[2].out_channels==co
        x=torch.randn(2,ci,7,9);got=m(x)
        expected=F.relu(F.conv2d(F.relu(F.conv2d(x,m[0].weight,padding=1)),m[2].weight,padding=1))
        close(got,expected);assert got.shape==(2,co,7,9)''',
    '예: (2,1,7,9)→Conv(1,3)→(2,3,7,9)→Conv(3,2)→(2,2,7,9)。kernel3,padding1,stride1이므로 공간 크기는 유지됩니다. U-Net 이름만 보고 BN을 추가하지 않습니다.',
    ('padding=1','padding=0'))

add('U03', 'Down: pooling 후 convolution', [7], '레이어 구성·동작',
    'Down.forward입니다. MaxPool2d(2) 후 제공 conv를 적용하세요. conv는 공간 크기를 유지하고 채널만 변경합니다.',
    '''def solve(x,conv):
    x=<<F.max_pool2d(x,kernel_size=2)>>
    return <<conv(x)>>''',
    '''def check(fn):
    for h,w in [(6,8),(7,9)]:
        x=torch.arange(h*w).float().reshape(1,1,h,w);conv=nn.Conv2d(1,3,1)
        expected=torch.stack([torch.stack([x[:,:,y:y+2,z:z+2].amax((-2,-1)) for z in range(0,w-1,2)],-1) for y in range(0,h-1,2)],-2)
        close(fn(x,conv),conv(expected));assert fn(x,conv).shape==(1,3,h//2,w//2)''',
    '예: (1,1,7,9)→MaxPool2d2→(1,1,3,4)→conv→(1,3,3,4)。홀수 크기는 내림되므로 나중에 Up에서 padding이 필요할 수 있습니다.',
    ('F.max_pool2d','F.avg_pool2d'))

add('U04', '두 업샘플링 방식', [7], '레이어 구성·동작',
    'bilinear=True이면 Upsample(scale_factor=2,mode="bilinear",align_corners=True), False이면 ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)를 반환하세요.',
    '''def solve(in_channels,bilinear):
    if bilinear:
        return <<nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)>>
    return <<nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)>>''',
    '''def check(fn):
    for c in (4,6):
        x=torch.randn(2,c,3,5)
        up=fn(c,True);assert isinstance(up,nn.Upsample)
        close(up(x),F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True))
        assert up(x).shape==(2,c,6,10) and len(list(up.parameters()))==0
        deconv=fn(c,False);assert isinstance(deconv,nn.ConvTranspose2d)
        assert deconv.kernel_size==(2,2) and deconv.stride==(2,2)
        assert deconv(x).shape==(2,c//2,6,10)''',
    '예: (2,4,3,5)→bilinear이면(2,4,6,10),transposed conv이면(2,2,6,10)。전자는 채널 유지·학습 파라미터 없음, 후자는 이 설정에서 채널 절반·학습 가능한 weight가 있습니다.',
    ('stride=2','stride=1'))

add('U05', 'Up: 크기 정렬과 skip concat', [7], '레이어 구성·동작',
    '업샘플링은 이미 완료된 x1=(B,Cup,h,w)를 제공합니다. x2=(B,Cskip,H,W)는 skip입니다. 부족한 크기를 좌우/상하에 나누어 padding하고 [x2,x1] 순서로 채널축 cat 후 제공 conv에 넣으세요. diff는0 이상입니다.',
    '''def solve(x1,x2,conv):
    dy=x2.size(2)-x1.size(2)
    dx=x2.size(3)-x1.size(3)
    x1=<<F.pad(x1,[dx//2,dx-dx//2,dy//2,dy-dy//2])>>
    x=<<torch.cat([x2,x1],dim=1)>>
    return conv(x)''',
    '''def check(fn):
    for dy,dx in [(1,3),(0,0),(2,1)]:
        x1=torch.arange(2*3*4).float().reshape(1,2,3,4);x2=torch.full((1,3,3+dy,4+dx),-1.)
        padded=torch.zeros(1,2,3+dy,4+dx)
        padded[:,:,dy//2:dy//2+3,dx//2:dx//2+4]=x1
        expected=torch.cat([x2,padded],1)
        close(fn(x1,x2,nn.Identity()),expected)
        conv=nn.Conv2d(5,2,1);close(fn(x1,x2,conv),conv(expected))''',
    '예: x1(1,2,3,4),skip(1,3,4,7)→padding은[왼1,오른2,위0,아래1]→cat(1,5,4,7)。ResNet skip은 덧셈, 여기의 U-Net skip은 채널 연결입니다.',
    ('torch.cat([x2,x1],dim=1)','torch.cat([x1,x2],dim=1)'))

add('U06', 'Decoder에 올바른 skip 전달', [7], '레이어 구성·동작',
    'Encoder의 x1,x2,x3,x4,x5가 이미 계산되어 있습니다. model.up1~up4와 outc를 이용해 bottleneck x5에서 깊은 skip부터 연결하세요. 전체 encoder를 재작성하지 않습니다.',
    '''def solve(model,x1,x2,x3,x4,x5):
    x=model.up1(x5,<<x4>>)
    x=model.up2(x,<<x3>>)
    x=model.up3(x,<<x2>>)
    x=model.up4(x,<<x1>>)
    return <<model.outc(x)>>''',
    '''def check(fn):
    class Join(nn.Module):
        def forward(self,a,b):
            assert b.shape[-2]==a.shape[-2]*2
            return F.interpolate(a,size=b.shape[-2:],mode='nearest')+b
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__();self.up1=Join();self.up2=Join();self.up3=Join();self.up4=Join();self.outc=nn.Conv2d(1,2,1)
    for base in (1.,2.):
        xs=[torch.full((1,1,s,s),base*(i+1)) for i,s in enumerate([16,8,4,2,1])]
        m=Decoder();got=fn(m,*xs)
        close(got,m.outc(torch.full((1,1,16,16),base*15)))''',
    '예: x5의1×1→up1+skip x4의2×2→x3의4×4→x2의8×8→x1의16×16。테스트 Join은 연결 순서만 확인하는 작은 대역이며 실제 U-Net 대신 정확도를 평가하지 않습니다. 실제 concat은 앞 문제에서 검증합니다.',
    ('model.up1(x5,x4)','model.up1(x5,x3)'))

add('U07', '픽셀 logits와 정답 mask로 학습', [9,11], '예측·정답을 학습으로 연결',
    'binary segmentation 한 배치를 학습하세요. logits와 masks는 모두 (B,1,H,W), masks는 float0/1입니다. BCEWithLogitsLoss를 선택하고 forward→loss→backward→step을 완성하세요. loss 전에 sigmoid를 붙이지 않습니다.',
    '''def solve(model,optimizer,images,masks):
    model.train()
    optimizer.zero_grad()
    criterion=<<nn.BCEWithLogitsLoss()>>
    logits=model(images)
    loss=<<criterion(logits,masks)>>
    <<loss.backward()>>
    <<optimizer.step()>>
    return loss.item()''',
    '''def check(fn):
    for seed in (5,9):
        torch.manual_seed(seed);a=nn.Conv2d(1,1,1);b=copy.deepcopy(a)
        oa=torch.optim.SGD(a.parameters(),lr=.05);ob=torch.optim.SGD(b.parameters(),lr=.05)
        for h,w in [(3,4),(4,5)]:
            x=torch.randn(2,1,h,w);target=(x>0).float()
            ob.zero_grad();loss=F.binary_cross_entropy_with_logits(b(x),target);loss.backward();ob.step()
            assert abs(fn(a,oa,x,target)-loss.item())<1e-6
            for p,q in zip(a.parameters(),b.parameters()):close(p,q)''',
    '예: 한 픽셀 logit0→확률.5,정답1이면 loss≈.693。이 연산을 모든 픽셀에 적용합니다. 이미지 분류의 정답(B,)과 달리 분할 정답은 픽셀마다 있습니다.',
    ('criterion(logits,masks)','criterion(logits.sigmoid(),masks)'))

add('U08', '확률을 binary segmentation mask로', [15], '모델 출력 활용·데이터 구성',
    'logits=(B,1,H,W)에 sigmoid 후 threshold 초과인 픽셀을1,나머지를0으로 바꿔 float 텐서로 반환하세요. 원본의0.5를 함수 인자로 일반화했습니다.',
    '''def solve(logits,threshold):
    probs=<<torch.sigmoid(logits)>>
    return <<(probs > threshold).float()>>''',
    '''def check(fn):
    probs=torch.tensor([[[[.1,.5,.8],[.6,.2,.95]]]])
    logits=torch.logit(probs)
    for threshold in (.5,.75):
        got=fn(logits,threshold)
        close(got,(probs>threshold).float())
        assert got.shape==(1,1,2,3) and got.dtype==torch.float32''',
    '예: 확률[.2,.5,.8],threshold=.5→mask[0,0,1]。원본은 >=가 아니라 >입니다. 한 채널 sigmoid이므로 채널축 argmax를 하면 모든 픽셀이0이 되는 잘못된 결과가 나옵니다.',
    ('probs > threshold','probs >= threshold'))

INTRO = '''# Vision · 범위 제한 기반 빈칸·부분 구현 문제

원본에 마스킹이 없어 **완성된 실습 코드를 바탕으로 새로 선정한 연습 문제**입니다. 실제 시험 문항이나 공식 배점을 의미하지 않습니다.

- ResNet: **0~5절**, 전체 셀15까지. 이번 요청의 “Chapter 6 이전”을 적용하여 6절 자체도 제외합니다.
- ViT: **6절 시작 전**, 전체 셀23까지. 원본 절3/4 순서가 뒤섞여 있어 셀 번호도 함께 기록합니다.
- DETR: 기존 확정 범위인 **13절 포함**, 전체 셀30까지.
- U-Net: 지정 노트북 전체. 설치·다운로드·긴 학습·그래프 꾸미기는 제외합니다.

근거: [시험 출제 포인트 PDF](../../AI인증시험_출제포인트.pdf) 1쪽 Vision의 레이어 이해, 예측/정답을 학습으로 연결, 데이터 구성, 이미지 변환 라이브러리, 평가 Metric. 문항별 셀 번호는 **Markdown 포함,1부터**입니다.

**사용법:** 준비 셀을 먼저 실행하고 각 문제에서 `MISSING` 또는 `raise NotImplementedError`를 올바른 코드로 교체 → 바로 아래 테스트 실행 → 실패하면 shape/축/순서 확인 → 마지막의 접힌 해설 확인. `# ????`는 원본에서 가져온 빈칸이 아니라 새 마스킹 표시입니다. 각 문제는 독립되어 이전 문제의 답이 필요 없습니다. 답을 수정하면 함수 셀을 먼저 다시 실행하세요. 예상 5~10분/문제(연습용 가안). 작성한 이 노트북을 제출하면 채점할 수 있습니다.

CPU·PyTorch·torchvision·einops·NumPy·Pillow만 사용합니다. 모델/데이터 다운로드 없음. 테스트는 shape뿐 아니라 수치, 학습 갱신, 잘못된 축, 다른 인자도 검사합니다. 통과는 작은 입력 검증이며 실전 정확도나 독립 숙달을 보장하지 않습니다.

**원본과 차이:** ViT FFN은 실제 코드의 ReLU 사용(주석의 GELU와 다름). U-Net DoubleConv는 실제 코드처럼 BN 없음. DETR attention reshape는 query/key 축이 뒤바뀐 설명·슬라이스를 교정하고 문제에서 명시합니다. ResNet AMP는 CPU에서 비활성 scaler로 순서를 확인합니다. 설치된 PyTorch 2.5.1/torchvision 0.20.1/einops 0.8.2에서 검증합니다.

이미지 텐서는 PyTorch의 실제 축 순서 **B,C,H,W**로 표기합니다. W,H를 구분하는 비정사각형 입력도 검사합니다. 이해가 안 되는 문제 번호·줄을 알려주면 작은 행렬 예시로 다시 설명할 수 있습니다.
'''

def build():
    report=[]
    cells=[nbformat.v4.new_markdown_cell(INTRO),nbformat.v4.new_code_cell(SETUP)]
    groups={key:[] for key in FILES}
    for i,s in enumerate(specs,1):
        group=s['key'][0]
        assert max(s['cells'])<=LIMITS[group]
        # All source references must point at code cells, within the allowed cutoff.
        original=json.loads((ROOT/'vision'/FILES[group]).read_text(encoding='utf-8'))
        assert all(original['cells'][j-1]['cell_type']=='code' for j in s['cells'])
        answer=re.sub(r'<<(.*?)>>',lambda m:m.group(1),s['code'],flags=re.S)
        def mask(m):
            val=m.group(1)
            start=s['code'].rfind('\n',0,m.start())+1
            standalone=not s['code'][start:m.start()].strip()
            if standalone and ('backward()' in val or 'step(' in val or 'update()' in val or 'zero_grad(' in val or 'load_state_dict' in val or 'handle.remove()' in val or 'model.eval()' in val):
                return 'raise NotImplementedError("???? 이 줄을 완성하세요")'
            return 'MISSING'
        blank=re.sub(r'<<(.*?)>>',mask,s['code'],flags=re.S)
        # Unique names make rerunning one test independent of later questions.
        name='solve_'+s['key'].lower()
        answer=answer.replace('def solve(',f'def {name}(')
        blank=blank.replace('def solve(',f'def {name}(')
        checkname='check_'+s['key'].lower()
        check=s['check'].replace('def check(',f'def {checkname}(')
        env={};exec(SETUP,env);exec(answer,env);exec(check,env)
        env[checkname](env[name])
        old,new=s['mutation'];assert old in answer,(s['key'],old)
        exec(answer.replace(old,new,1),env)
        try:env[checkname](env[name])
        except Exception:pass
        else:raise AssertionError('Mutant accepted: '+s['key'])
        exec(blank,env)
        try:env[checkname](env[name])
        except Exception:pass
        else:raise AssertionError('Blank accepted: '+s['key'])
        source=f'../../vision/{FILES[group]}'
        prompt=f'## {s["key"]}. {s["title"]}\n\n**PDF 점검:** {s["criteria"]} · **근거:** [{FILES[group]}]({source}) 전체 셀 {", ".join(map(str,s["cells"]))}\n\n{s["task"]}\n\n완료 기준: 아래 테스트에서 PASS. 해설을 닫고 입력→출력 shape와 빈칸 이유도 말로 설명해 보세요.'
        # Imports are visible in every answer cell instead of unexplained aliases.
        imports='import torch\nimport torch.nn as nn  # nn = torch.nn\nimport torch.nn.functional as F  # F = torch.nn.functional\n'
        if group in 'RD':imports+='import torchvision.transforms as T  # T = torchvision.transforms\nfrom torch.utils.data import DataLoader\n'
        if group=='V':imports+='from einops import rearrange, repeat\nfrom einops.layers.torch import Rearrange\n'
        if group=='U':imports+='import numpy as np\n'
        test=check+f'\n\ntry:\n    {checkname}({name})\n    print("PASS · {s["key"]}")\nexcept Exception as error:\n    print("FAIL · {s["key"]} ·",type(error).__name__,str(error))\n'
        explanation=f'<details><summary>시도 후 펼치기: 쉬운 설명 · 예시 · 정답</summary>\n\n{s["explain"]}\n\n```python\n{imports}\n{answer}\n```\n\n다음 복습: 해설 없이 다시 작성하고, 배치/채널/클래스 수가 바뀌어도 동작하는 이유를 설명하세요.\n</details>'
        block=[nbformat.v4.new_markdown_cell(prompt),nbformat.v4.new_code_cell(imports+'\n# ???? 표시: MISSING 또는 NotImplementedError를 교체하세요.\n'+blank),nbformat.v4.new_code_cell(test),nbformat.v4.new_markdown_cell(explanation)]
        cells.extend(block);groups[group].extend(block)
        report.append(dict(id=s['key'],title=s['title'],source=FILES[group],source_cells=s['cells'],criteria=s['criteria'],answer_pass=True,blank_rejected=True,mutation_rejected=True,mutation=[old,new]))
    metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},'language_info':{'name':'python'}}
    def write(name,items):
        book=nbformat.v4.new_notebook(cells=items,metadata=metadata)
        nbformat.validate(book);nbformat.write(book,HERE/name)
    write('vision-exercises.ipynb',cells)
    for key,name in [('R','01-resnet'),('V','02-vit'),('D','03-detr'),('U','04-unet')]:
        write(name+'-exercises.ipynb',[nbformat.v4.new_markdown_cell(INTRO),nbformat.v4.new_code_cell(SETUP)]+groups[key])
    (HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={name:hashlib.sha256((ROOT/'vision'/name).read_bytes()).hexdigest() for name in FILES.values()}
    (HERE/'source-manifest.json').write_text(json.dumps({'sources':hashes,'cutoffs_inclusive':LIMITS},indent=2),encoding='utf-8')
    mapping=['# PDF 출제 포인트·자가점검과 문제 대응','',
        '근거: `AI인증시험_출제포인트.pdf` 1쪽 Vision. 이 표는 연습문제 설계이며 실제 출제를 확정하지 않습니다.','',
        '| PDF 자가점검 항목 | 대응 문제 |','|---|---|',
        '| 주요 layer의 의미·구성·동작·활용 | R04, V01~V09, D04~D05, U02~U06 |',
        '| 모델 결과와 정답을 이용하여 학습으로 연결 | R05~R06, V10~V11, U07 |',
        '| 모델·학습·평가 의도에 맞는 데이터 구성 | R01~R03, V01~V02, D01~D03, U01·U08 |',
        '| torchvision.transforms 등의 이미지 변형 | R01·R03, D01 |',
        '| 평가 Metric 원리와 방법 | R07 정확도·샘플 평균 loss, R08 confidence와 정확도 구분 |','',
        '범위 안 평가 루프의 top-1 accuracy를 코드로 평가합니다. PDF는 구체적인 지표명을 지정하지 않습니다. ResNet6절 confusion matrix는 이번 범위 밖이고, DETR 실습에는 mAP·Hungarian loss 구현이 없어 코드 문제를 새로 추가하지 않았습니다. U-Net은 BCE loss/이진 예측을 다루며 loss를 IoU·Dice와 같은 지표로 혼동하지 않습니다.','',
        '| 문제 | 주제 | 원본 파일 | 전체 셀(1부터) | PDF 연결 |','|---|---|---|---|---|']
    mapping += [f'| {s["key"]} | {s["title"]} | {FILES[s["key"][0]]} | {", ".join(map(str,s["cells"]))} | {s["criteria"]} |' for s in specs]
    (HERE/'coverage.md').write_text('\n'.join(mapping)+'\n',encoding='utf-8')
    print(f'Generated {len(specs)} problems; answers, blanks, and mutants checked.')

if __name__=='__main__':
    build()
