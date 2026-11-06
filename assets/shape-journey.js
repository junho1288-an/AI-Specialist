"use strict";
(() => {
 const steps=[...document.querySelectorAll('.lesson-step')];
 if(!steps.length)return;
 const buttons=[...document.querySelectorAll('[data-step-go]')];
 let current=0;
 const show=i=>{
  current=Math.min(Math.max(i,0),steps.length-1);
  steps.forEach((s,j)=>s.hidden=j!==current);
  buttons.forEach((b,j)=>{if(j===current)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  document.getElementById('step-progress').textContent=`${current+1} / ${steps.length} 단계 · 이론 → shape·코드 문제 → 피드백 → 다음 단계`;
  document.getElementById('step-prev').disabled=current===0;
  document.getElementById('step-next').disabled=current===steps.length-1;
 };
 buttons.forEach(b=>b.addEventListener('click',()=>show(Number(b.dataset.stepGo))));
 document.getElementById('step-prev').addEventListener('click',()=>{show(current-1);document.getElementById('step-progress').scrollIntoView({behavior:'smooth'});});
 document.getElementById('step-next').addEventListener('click',()=>{show(current+1);document.getElementById('step-progress').scrollIntoView({behavior:'smooth'});});
 const svgNS='http://www.w3.org/2000/svg';
 document.querySelectorAll('[data-feature-stack]').forEach(svg=>{
  const channels=Number(svg.dataset.channels),size=Number(svg.dataset.spatial),name=svg.dataset.name;
  svg.setAttribute('viewBox','0 0 500 210');svg.setAttribute('role','img');
  const title=document.createElementNS(svgNS,'title');title.textContent=`${name}: 채널 ${channels}개, 각 ${size}×${size}, 이미지 한 장 기준`;svg.append(title);
  const side=Math.max(45,Math.min(125,size*3.5));
  for(let i=4;i>=0;i--){const r=document.createElementNS(svgNS,'rect');r.setAttribute('x',String(65+i*15));r.setAttribute('y',String(35+i*6));r.setAttribute('width',String(side));r.setAttribute('height',String(side));r.setAttribute('fill',i%2?'#c5e4d4':'#e3f1e9');r.setAttribute('stroke','#37715a');svg.append(r);}
  const text=(x,y,t)=>{const e=document.createElementNS(svgNS,'text');e.setAttribute('x',x);e.setAttribute('y',y);e.setAttribute('font-size','16');e.textContent=t;svg.append(e);};
  text('290','65',name);text('290','98',`${channels}개 채널`);text('290','130',`각 ${size} × ${size}`);text('40','194','그림의 겹 수는 생략 · 실제 개수는 오른쪽 숫자');
 });
 const input=document.getElementById('shape-batch');
 const update=()=>{const n=Number(input.value);if(!Number.isInteger(n)||n<1||n>256)return;document.querySelectorAll('[data-batch-shape]').forEach(e=>{e.textContent='('+[n,...e.dataset.batchShape.split(',')].join(',')+')';});};
 input.addEventListener('input',update);update();
 document.querySelectorAll('form[data-shape-answer]').forEach(form=>form.addEventListener('submit',event=>{
  event.preventDefault();const value=String(new FormData(form).get('answer')||'').replace(/[\s()[\]]/g,'');const correct=value===form.dataset.shapeAnswer;
  const out=form.querySelector('.feedback');out.className='feedback '+(correct?'correct':'incorrect');
  out.textContent=!value?'먼저 shape를 적어 보세요.':(correct?'정답입니다. ':'다시 확인해 보세요. ')+form.dataset.explanation;
 }));
 show(0);
})();
