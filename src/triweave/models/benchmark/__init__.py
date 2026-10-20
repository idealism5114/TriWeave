"""Versioned CMGM benchmark migration; no source-project runtime dependencies."""
from .baselines import DEEP_V3, Recurrent, VanillaTransformer, GraphWaveNet, MTGNN, ScalarGraph, TCN
from .ablations import CONTROLS, FORMAL, PHASE4, FUSION, NODE_HISTORY, SLBRControl
from .classical import CLASSICAL, MaskedClassical

BASELINES = DEEP_V3 + ('TCN',) + CLASSICAL
MODEL_IDS = tuple('baseline/'+name for name in BASELINES) + tuple('ablation/'+name for name in CONTROLS)


def make_model(model_id,assets,targets):
    if model_id not in MODEL_IDS:raise ValueError('Unregistered model: '+model_id)
    family,name=model_id.split('/')
    if family=='ablation':return SLBRControl(assets,targets,name)
    if name in CLASSICAL:return MaskedClassical(name,assets,targets)
    if name in ('RNN','GRU','LSTM'):return Recurrent(assets,targets,name)
    if name in ('MSGNet','CrossGNN'):return ScalarGraph(assets,targets,name)
    return {'VanillaTransformer':VanillaTransformer,'GraphWaveNet':GraphWaveNet,'MTGNN':MTGNN,'TCN':TCN}[name](assets,targets)


def registry():
    result=[]
    for model_id in MODEL_IDS:
        family,name=model_id.split('/')
        if family=='baseline':
            group='DeepV3_latest' if name in DEEP_V3 else 'supplemental_latest_TCN_repair' if name=='TCN' else 'supplemental_fixed_classical'
            source='cmgm/models/comparison_baselines.py' if name in DEEP_V3 else 'cmgm/models/formal_baselines_v2.py' if name=='TCN' else 'cmgm/scripts/fixed_baseline_protocol.py'
            prefix=None if name in CLASSICAL else name not in ('MSGNet','CrossGNN')
        else:
            group='formal_SLBR_controls' if name in FORMAL else 'phase4_simplification' if name in PHASE4 else 'fusion_controls' if name in FUSION else 'node_history_controls'
            source='cmgm/models/formal_d0b_ablation.py' if name in FORMAL else 'cmgm/models/phase4_variants.py' if name in PHASE4 else 'cmgm/models/node_history_readout.py' if name in NODE_HISTORY or name=='global_concat_fusion' else 'cmgm/models/phase5_variants.py' if name in ('concat_linear64','gated_concat128','mean_concat128') else 'cmgm/models/phase6_variants.py'
            prefix=True
        result.append(dict(model_id=model_id,family=family,group=group,source=source,
            source_variant=CONTROLS.get(name,name),input_assets=311,features=26,targets=23,horizons=[1,5,10,20],
            prefix_causal_encoder=prefix,origin_window_causal=True,trained_on_TriWeave=False))
    return result
