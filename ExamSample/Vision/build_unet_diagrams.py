"""Build two static SVG U-Net diagrams and verify shapes against course classes."""
import ast
import html
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
torch.set_num_threads(1)
book=json.loads((ROOT/'vision/04_Unet.ipynb').read_text(encoding='utf-8'))
code=next(''.join(c['source']) for c in book['cells'] if 'class UNet(' in ''.join(c['source']))
tree=ast.parse(code)
classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'DoubleConv','Down','Up','OutConv','UNet'}]
env={'torch':torch,'nn':nn,'F':F}
exec(compile(ast.Module(body=classes,type_ignores=[]),'course-unet','exec'),env)
reports={}
for flag in (True,False):
    model=env['UNet'](n_channels=1,n_classes=1,bilinear=flag).eval()
    shapes={}
    hooks=[]
    for name in ['inc','down1','down2','down3','down4','up1','up2','up3','up4','outc']:
        module=getattr(model,name)
        def hook(module,inputs,output,key=name):shapes[key]=list(output.shape)
        hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():model(torch.zeros(1,1,128,128))
    for h in hooks:h.remove()
    reports[str(flag)]=shapes
    del model

def diagram(flag):
    factor=2 if flag else 1
    enc=[64,128,256,512,1024//factor]
    outs=[64,128//factor,256//factor,512//factor]
    mids=[64,128,256,512] if flag else outs
    inputs=[128,256,512,1024]
    sizes=[128,64,32,16,8]
    ys=[100,285,470,655,840]
    a=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1360 1030" role="img" aria-label="U-Net '+str(flag)+'의 채널과 공간 크기"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker></defs>']
    def text(x,y,s,size=15,color='#203247',anchor='middle'):
        a.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" font-size="{size}" fill="{color}">{html.escape(str(s))}</text>')
    def arrow(x,y,xx,yy,color='#446489'):
        a.append(f'<path d="M{x} {y} L{xx} {yy}" fill="none" stroke="{color}" stroke-width="2.5" marker-end="url(#arr)"/>')
    def block(x,y,c,hw,label,color):
        # All blocks are schematic: labels, not physical width, encode exact dimensions.
        for d in [12,6,0]:
            a.append(f'<rect x="{x+d}" y="{y-d/2}" width="72" height="64" rx="2" fill="{color}" stroke="#54708b"/>')
        text(x+36,y-18,f'{label}: {c}',16)
        text(x+36,y+87,f'{hw}×{hw}',15)
    text(210,30,'Encoder · 공간 ↓ / 채널 ↑',23)
    text(990,30,'Decoder · 공간 ↑ / 채널 조정',23)
    text(600,30,'회색 화살표 = skip 복사',16,'#687888')
    for i in range(5):
        y=ys[i];sz=sizes[i];out=enc[i];incoming=1 if i==0 else enc[i-1]
        block(40,y,incoming,sz,'IN','#c5dcf3')
        block(210,y,out,sz,'MID','#ffe1a5')
        block(380,y,out,sz,'OUT','#accff0')
        arrow(126,y+32,200,y+32);arrow(296,y+32,370,y+32)
        text(163,y+60,'Conv+ReLU',11);text(333,y+60,'Conv+ReLU',11)
        text(245,y+115,('inc' if i==0 else f'down{i}')+f' → x{i+1} = (1,{out},{sz},{sz})',17)
        if i<4:
            a.append(f'<path d="M418 {y+123} L418 {y+143} L76 {y+143} L76 {ys[i+1]-45}" fill="none" stroke="#bf4444" stroke-width="2.5" marker-end="url(#arr)"/>')
            text(245,y+163,'MaxPool2d(2): H,W 절반',13,'#a13333')
    for i in range(4):
        y=ys[i];sz=sizes[i];inc=inputs[i];mid=mids[i];out=outs[i];skip=enc[i]
        block(820,y,inc,sz,'IN(cat)','#b4e3d7')
        block(1000,y,mid,sz,'MID','#ffe1a5')
        block(1180,y,out,sz,'OUT','#c5b7eb')
        arrow(912,y+32,990,y+32);arrow(1092,y+32,1170,y+32)
        text(951,y+60,'Conv+ReLU',11);text(1131,y+60,'Conv+ReLU',11)
        arrow(470,y+25,810,y+25,'#88949f')
        text(640,y+8,f'x{i+1} skip: {skip}채널',15,'#626f7c')
        text(640,y+52,f'{skip} + {skip} = {inc}',17,'#19785d')
        text(1040,y+115,f'up{4-i} 출력 = (1,{out},{sz},{sz})',17)
        # Output of next-deeper stage rises into this stage, then concatenates.
        src_c=enc[4] if i==3 else outs[i+1]
        src_y=ys[i+1]
        src_x=418 if i==3 else 1216
        bend_y=y+153
        a.append(f'<path d="M{src_x} {src_y-42} L{src_x} {bend_y+20} L740 {bend_y+20} L740 {y+75} L810 {y+75}" fill="none" stroke="#239976" stroke-width="2.5" marker-end="url(#arr)"/>')
        label=f'Upsample: {src_c}→{skip} 채널 유지' if flag else f'ConvTranspose2d: {src_c}→{skip} 채널'
        text(1030,bend_y,label,14,'#167253')
        text(665,y+100,f'{sz//2}×{sz//2} → {sz}×{sz}',13,'#167253')
    # Final output head placed above the last decoder output.
    arrow(1216,72,1216,50,'#8055b0')
    text(1216,43,'1×1 Conv → 1채널 logits',12,'#654091')
    text(1120,1000,'최종: up4의 64채널 → OutConv(1×1, 64→1) → logits (1,1,128,128)',17,'#654091',anchor='end')
    text(675,945,f'Bottleneck x5: (1,{enc[4]},8,8) · factor={factor}',20)
    a.append('</svg>')
    # Match all annotated stage outputs to actual forward hooks.
    report=reports[str(flag)]
    for i in range(5):assert report['inc' if i==0 else f'down{i}']==[1,enc[i],sizes[i],sizes[i]]
    for i in range(4):assert report[f'up{4-i}']==[1,outs[i],sizes[i],sizes[i]]
    assert report['outc']==[1,1,128,128]
    return ''.join(a)

parts=[]
for flag in (True,False):
    svg=diagram(flag)
    name='unet-bilinear-'+str(flag).lower()+'.svg'
    (HERE/name).write_text(svg,encoding='utf-8')
    parts.append(f'<section id="{str(flag).lower()}"><h2>bilinear={flag} · factor={2 if flag else 1}</h2><p>'+('학습 없는 bilinear 확대: 채널 유지. DoubleConv의 MID는 IN//2입니다.' if flag else '학습 가능한 전치합성곱: 채널 절반. DoubleConv의 MID는 OUT과 같습니다.')+f'</p><a href="{name}">그림만 크게 열기 / 저장</a><div class="diagram">{svg}</div></section>')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실습 U-Net 전체 구조: bilinear 비교</title><style>
body{margin:0;background:#f3f6fa;color:#203247;font:17px/1.65 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:26px}h1{font-size:30px}section{background:white;border:1px solid #dbe3ed;border-radius:14px;padding:22px;margin:28px 0}h2{margin-top:0}.diagram{overflow-x:auto}.diagram svg{width:100%;min-width:1050px;height:auto}code{background:#e6edf5;padding:3px 6px}.note{padding:18px;background:#fff0cf;border-radius:12px}a{color:#235fa5}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dbe3ed;text-align:left}@media print{section{break-before:page}.diagram svg{min-width:0}body{background:white}}</style><main>
<h1>U-Net 전체 구조 · 두 그림으로 비교하기</h1>
<p><a href="#true">True 그림</a> · <a href="#false">False 그림</a> · <a href="../../vision/04_Unet.ipynb">기준 실습 코드</a></p>
<div class="note"><strong>입력 (B,C,H,W)=(1,1,128,128), n_classes=1.</strong> 블록 위 숫자는 채널 수, 아래 숫자는 H×W입니다. 블록 크기·겹친 장수는 개념 표현이며 실제 채널 수는 숫자를 읽으세요.<br>IN → Conv3×3+ReLU → MID → Conv3×3+ReLU → OUT. <strong>MID는 별도 레이어가 아니라 두 Conv 사이 텐서의 채널 수</strong>입니다.</div>
<p>왼쪽은 위→아래로, 오른쪽은 아래→위로 읽으세요. 초록 선은 업샘플링, 회색 선은 encoder의 skip을 복사해 <strong>채널축 concat</strong>하는 경로입니다. Decoder IN(cat)은 이미 업샘플 결과와 skip을 합친 상태입니다.</p>
'''+''.join(parts)+'''
<section><h2>가장 헷갈리는 up1 한 줄</h2><table><tr><th></th><th>True</th><th>False</th></tr><tr><td>x5</td><td>512채널,8×8</td><td>1024채널,8×8</td></tr><tr><td>업샘플 후</td><td>512채널,16×16</td><td>512채널,16×16</td></tr><tr><td>skip x4와 concat</td><td>512+512=1024채널</td><td>512+512=1024채널</td></tr><tr><td>DoubleConv</td><td>IN1024 → MID512 → OUT256</td><td>IN1024 → MID512 → OUT512</td></tr></table><p>따라서 True에서도 <code>Up(1024,256)</code>은 맞습니다. 1024는 x5 단독 채널이 아니라 concat 뒤 채널입니다.</p></section>
<p><strong>교안과 차이:</strong> 현재 PDF18쪽은 Faster R-CNN anchor 그림이고, U-Net 구조는81쪽입니다. 그 U자 표현을 참고하되 실제 실습 코드의 padding=1을 적용했습니다. 그래서 Conv마다 공간 크기가 줄지 않습니다. 이 예시는128이16으로 나누어져 모든 skip 크기가 맞으며 F.pad의 추가량은0입니다.</p>
<p>OutConv는1×1 Conv이며 sigmoid 없이 logits를 반환합니다. n_classes가 K이면 마지막 출력만 (1,K,128,128)입니다. U-Net DoubleConv에는 이 실습 코드 기준 BN이 없습니다.</p>
<p>검증: 원본 노트북의 모델 클래스만 추출해 두 설정을 CPU에서 forward하고 각 단계 shape를 대조했습니다. 데이터 다운로드·학습 없음. 질문할 때 x1~x5 또는 up1~up4를 알려주시면 해당 구간을 확대해 설명할 수 있습니다.</p></main></html>'''
(HERE/'unet-full-architecture.html').write_text(page,encoding='utf-8')
(HERE/'unet-diagram-validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print('Two SVG diagrams and HTML generated; both actual model forward shape checks passed.')
