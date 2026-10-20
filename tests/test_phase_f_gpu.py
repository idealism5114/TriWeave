"""Continuation restores optimizer/scheduler/RNG, not just old model weights."""
import copy
import pandas as pd
import pytest
import torch
from triweave.phase_f_gpu import history_prefix, restore_training_state


def test_history_drops_unsaved_tail_only_in_new_branch():
    history=pd.DataFrame(dict(epoch=[1,2,3,4],score=[4.,3.,2.,3.]))
    prefix=history_prefix(history,3)
    assert prefix.epoch.tolist()==[1,2,3] and len(history)==4
    with pytest.raises(RuntimeError):history_prefix(history.loc[history.epoch!=2],3)


def test_optimizer_scheduler_rng_exact_CPU_continuation():
    torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.3),torch.nn.Linear(3,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5)
    x=torch.ones(4,2)
    model(x).square().mean().backward();optimizer.step();scheduler.step(.1)
    payload=copy.deepcopy(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),rng_state=torch.get_rng_state()))
    optimizer.zero_grad();expected=model(x);expected.square().mean().backward();optimizer.step()
    weights=copy.deepcopy(model.state_dict())
    recovered=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.3),torch.nn.Linear(3,1))
    opt=torch.optim.Adam(recovered.parameters(),lr=.9)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=1)
    restore_training_state(recovered,opt,sched,payload,torch.device('cpu'))
    actual=recovered(x);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    opt.zero_grad();actual.square().mean().backward();opt.step()
    for name,value in recovered.state_dict().items():torch.testing.assert_close(value,weights[name],rtol=0,atol=0)
    assert sched.state_dict()==payload['scheduler_state_dict']
