"""Runner gates/selection with a synthetic fixture and optimizer updates disabled."""
import json
from pathlib import Path
import types
import pandas as pd
import pytest
import torch
from triweave import benchmark as runner


def test_cli_does_not_train_implicitly(monkeypatch):
    monkeypatch.setattr('sys.argv',['triweave.benchmark','--train'])
    with pytest.raises(SystemExit) as error:runner.main()
    assert error.value.code==2


def test_one_run_uses_macro_score_and_restores_true_argmin_without_optimizer_updates(tmp_path,monkeypatch):
    output=tmp_path/runner.OUT;output.mkdir(parents=True)
    (output/'preflight.json').write_text(json.dumps(dict(PASS=True,source_hashes={})))
    data=tmp_path/runner.DATA;data.mkdir(parents=True);(data/'dataset_manifest.json').write_text('{}')
    monkeypatch.setattr(runner,'hashes',lambda root:{})
    monkeypatch.setattr(runner,'dataset_gate',lambda *args:None)
    monkeypatch.setattr(runner,'manifests',lambda root:([dict(asset_id='fixture')],['fixture']))
    monkeypatch.setattr(runner.phase,'seed42',lambda:torch.manual_seed(42))
    monkeypatch.setattr(runner.phase,'environment',lambda:dict(fixture=True))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(4,23));self.assets=[dict(asset_id='fixture')]
        def forward(self,**kwargs):return self.p[None]
        def set_epoch(self,epoch):pass
        def switch_loss(self):return self.p.sum()*0.
    monkeypatch.setattr(runner,'make_model',lambda *args:Model())
    batch={k:torch.zeros(1,1) for k in runner.INPUT_KEYS}
    batch.update(Y=torch.ones(1,4,23),label_valid_mask=torch.ones(1,4,23,dtype=torch.bool),origin_date=['synthetic'])
    monkeypatch.setattr(runner.phase,'loaders',lambda *args:{'TRAIN':[batch],'VAL':[batch]})
    scores=[.3,.1]+[.2]*10+[.1]
    monkeypatch.setattr(runner.phase,'evaluate',lambda *args:dict(metrics=dict(VAL_SELECTION_SCORE=scores.pop(0))))
    monkeypatch.setattr(runner,'save_validation',lambda *args:None)
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *args,**kwargs:None)
    runner.train(tmp_path,'baseline/RNN','cpu','fixture')
    out=output/'runs/fixture'
    complete=json.loads((out/'training_complete.json').read_text())
    history=pd.read_csv(out/'training_history.csv')
    assert complete['best_epoch']==2 and len(history)==12 and not scores
    cp=torch.load(out/'best.pt',weights_only=False)
    assert torch.count_nonzero(cp['model_state_dict']['p'])==0
    assert json.loads((out/'checkpoint_audit.json').read_text())['PASS']
