"""Supplemental fixed CMGM estimators, grouping heads by TRAIN label masks."""
import numpy as np
import torch
from .common import sanitize

CLASSICAL = ('Ridge','RandomForest','XGBoost')


class MaskedClassical:
    def __init__(self,name,assets,targets,seed=42,jobs=4):
        if name not in CLASSICAL:raise ValueError(name)
        self.name=name;self.assets=assets;self.targets=targets;self.seed=seed;self.jobs=jobs
        self.groups=[];self.fitted_outputs=np.zeros(4*len(targets),bool)

    def configuration(self):
        if self.name=='Ridge':return dict(alpha=1.,fit_intercept=True,solver='lsqr')
        if self.name=='RandomForest':return dict(n_estimators=300,max_depth=20,max_features='sqrt',
            min_samples_split=2,min_samples_leaf=1,bootstrap=True,random_state=self.seed,n_jobs=self.jobs)
        return dict(n_estimators=200,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,
            objective='reg:squarederror',reg_lambda=1.,reg_alpha=0.,tree_method='hist',random_state=self.seed,n_jobs=self.jobs)

    def make_estimator(self):
        if self.name=='Ridge':
            from sklearn.linear_model import Ridge
            return Ridge(**self.configuration())
        if self.name=='RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.configuration())
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        return MultiOutputRegressor(XGBRegressor(**self.configuration()),n_jobs=1)

    @staticmethod
    def design(X,input_valid_mask,feature_valid_mask,feature_applicable_mask):
        tensors=[torch.as_tensor(x) for x in (X,input_valid_mask,feature_valid_mask,feature_applicable_mask)]
        x,_=sanitize(*tensors)
        return x.detach().cpu().numpy().reshape(len(x),-1)

    def fit(self,X,Y,input_valid_mask,feature_valid_mask,feature_applicable_mask,label_valid_mask,*,split):
        if split!='TRAIN':raise ValueError('Classical fitting permits TRAIN only')
        x=self.design(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        return self.fit_design(x,Y,label_valid_mask,split=split)

    def fit_design(self,x,Y,label_valid_mask,*,split):
        if split!='TRAIN':raise ValueError('Classical fitting permits TRAIN only')
        if self.groups:raise RuntimeError('Already fitted; no implicit refit')
        if x.ndim!=2 or x.shape[1]!=20*len(self.assets)*26:raise ValueError('Full input layout mismatch')
        y=np.asarray(Y).reshape(len(x),-1);mask=np.asarray(label_valid_mask,dtype=bool).reshape(y.shape)
        if y.shape[1]!=4*len(self.targets):raise ValueError('Target shape/order mismatch')
        grouped={}
        for j in range(y.shape[1]):
            if mask[:,j].any():grouped.setdefault(mask[:,j].tobytes(),[]).append(j)
        for indices in grouped.values():
            rows=mask[:,indices[0]]
            if not np.isfinite(y[rows][:,indices]).all():raise ValueError('Nonfinite valid TRAIN label')
            estimator=self.make_estimator();estimator.fit(x[rows],y[rows][:,indices])
            self.groups.append((indices,estimator));self.fitted_outputs[indices]=True
        return self

    def predict(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,*,return_mask=False):
        x=self.design(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        return self.predict_design(x,return_mask=return_mask)

    def predict_design(self,x,*,return_mask=False):
        if not self.groups:raise RuntimeError('Estimator not fitted')
        if not return_mask and not self.fitted_outputs.all():raise RuntimeError('Some heads have no TRAIN labels; request return_mask=True')
        result=np.zeros((len(x),len(self.fitted_outputs)),np.float64)
        for indices,estimator in self.groups:result[:,indices]=np.asarray(estimator.predict(x)).reshape(len(x),len(indices))
        result=result.reshape(len(x),4,len(self.targets))
        mask=np.broadcast_to(self.fitted_outputs.reshape(1,4,-1),result.shape).copy()
        if not np.isfinite(result[mask]).all():raise FloatingPointError('Nonfinite classical prediction')
        return (result,mask) if return_mask else result
