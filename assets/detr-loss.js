"use strict";
(() => {
 const costs = [[8,1],[1,8],[2,7]];
 const cat = document.getElementById("cat-choice");
 const dog = document.getElementById("dog-choice");
 const result = document.getElementById("matching-result");
 if (!cat || !dog || !result) return;
 const update = () => {
  const a=Number(cat.value),b=Number(dog.value);
  if(a===b){result.textContent="같은 예측을 두 정답에 배정할 수 없습니다. 서로 다른 예측을 골라 주세요.";return;}
  const total=costs[a][0]+costs[b][1];
  const remaining=[0,1,2].find(i=>i!==a&&i!==b);
  result.textContent=`P${a+1}→고양이, P${b+1}→강아지: 비용 ${costs[a][0]} + ${costs[b][1]} = ${total}. `+(total===2?"이 표의 최소 비용입니다. ":"더 작은 합을 찾아보세요. ")+`남은 P${remaining+1}은 no-object 학습 대상입니다.`;
 };
 cat.addEventListener("change",update);dog.addEventListener("change",update);
 document.getElementById("matching-best").addEventListener("click",()=>{cat.value="1";dog.value="0";update();});
 update();
})();
