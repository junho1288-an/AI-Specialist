"""Check each reference answer in isolation and reject representative wrong answers."""
import ast, contextlib, io, json, re
from pathlib import Path
import nbformat

folder=Path(__file__).parent
book=nbformat.read(folder/'llm-exercises.ipynb',4)
mutations={
    2:('enc_sample[1:context_size+1]','enc_sample[0:context_size]'),
    6:('-torch.inf','0.0'),
    8:('unbiased=False','unbiased=True'),
    12:('logits[:, -1, :]','logits[:, 0, :]'),
    14:('optimizer.zero_grad()','pass'),
    21:('[:, -1, :]','[:, 0, :]'),
    23:('(self.alpha / self.rank)','(1.0)'),
    24:('self.linear(x) + self.lora(x)','self.linear(x)'),
}
report=[]
for q in range(1,27):
    offset=2+(q-1)*4
    answer=re.findall(r'```python\n(.*?)\n```',book.cells[offset+3].source,re.S)[0]
    env={}
    exec(book.cells[1].source,env)
    exec(answer,env)
    # Avoid the student-facing exception handler: an assertion must fail this validator.
    tree=ast.parse(book.cells[offset+2].source)
    checker=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    exec(compile(ast.Module(body=[checker],type_ignores=[]),'test','exec'),env)
    with contextlib.redirect_stdout(io.StringIO()):env[checker.name]()
    row={'question':q,'independent_reference_pass':True}
    if q in mutations:
        old,new=mutations[q]
        assert old in answer,(q,old)
        exec(answer.replace(old,new,1),env)
        try:
            with contextlib.redirect_stdout(io.StringIO()):env[checker.name]()
        except Exception:row['semantic_mutation_rejected']=True
        else:raise AssertionError(('incorrect answer accepted',q))
    report.append(row)
(folder/'mutation-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('26 isolated references passed; 8 semantic mutations rejected')
