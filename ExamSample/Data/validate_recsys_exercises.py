"""Validate notebook answers, blank state, and representative wrong implementations."""
import ast,asyncio,contextlib,io,json,re
from pathlib import Path
import nbformat
from nbclient import NotebookClient

folder=Path(__file__).parent
book=nbformat.read(folder/'recsys-gcf-exercises.ipynb',4)
mutations={
 1:('torch.sqrt(deg[src] * deg[dst])','(deg[src] * deg[dst])'),
 2:('self.W1(src_feat)','self.W1(dst_feat)'),
 3:('index_add_(','index_add('),
 4:('self.num_users,self.num_items','self.num_items,self.num_users'),
 5:('dim=-1','dim=0'),
}
results=[]
for q in range(1,6):
 offset=2+(q-1)*4
 answer=re.findall(r'```python\n(.*?)\n```',book.cells[offset+3].source,re.S)[0]
 env={};exec(book.cells[1].source,env);exec(answer,env)
 checker=next(n for n in ast.parse(book.cells[offset+2].source).body if isinstance(n,ast.FunctionDef))
 exec(compile(ast.Module(body=[checker],type_ignores=[]),'test','exec'),env)
 with contextlib.redirect_stdout(io.StringIO()):env[checker.name]()
 row={'question':q,'independent_answer_pass':True}
 if q in mutations:
  old,new=mutations[q];assert old in answer,(q,old)
  exec(answer.replace(old,new,1),env)
  try:
   with contextlib.redirect_stdout(io.StringIO()):env[checker.name]()
  except Exception:row['semantic_mutation_rejected']=True
  else:raise AssertionError(('wrong answer accepted',q))
 results.append(row)
print('5 independent answers passed; 5 wrong implementations rejected')
if hasattr(asyncio,'WindowsSelectorEventLoopPolicy'):
 asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
for mode in ['blank','answer']:
 n=nbformat.read(folder/'recsys-gcf-exercises.ipynb',4)
 if mode=='answer':
  for i in range(2,len(n.cells),4):n.cells[i+1].source=re.findall(r'```python\n(.*?)\n```',n.cells[i+3].source,re.S)[0]
 NotebookClient(n,timeout=120,kernel_name='python3').execute()
 output=''.join(o.get('text','') for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]))
 assert output.count('PASS' if mode=='answer' else 'FAIL')==5,output
 assert output.count('FAIL' if mode=='answer' else 'PASS')==0,output
 print(mode,'fresh kernel checked',5)
(folder/'recsys-kernel-validation.json').write_text(json.dumps({'fresh_answer_pass':5,'fresh_blank_fail':5,'isolated_checks':results},indent=2),encoding='utf-8')
