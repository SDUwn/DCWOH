%% init the workspace
close all; clear; clc; warning off;
addpath(genpath('./vlfeat/toolbox/'));
addpath(genpath('./Method/'));
addpath(genpath('./util/'));
run vl_setup;
%%
train_param.ds_name='NUSWIDE-clip';  
train_param.N=40000;
train_param.chunk_size=5000; 
train_param.query_size=2000;
%%
train_param.ds_name='MIRFlickr-clip'; 
train_param.N=16000;
train_param.chunk_size=2000;
train_param.query_size=1000;
%%
train_param.normalizeX = 0;
train_param.kernel = 0;
train_param.unsupervised=0;
[train_param,XTrain,LTrain,LLTrain,HLTrain,YTrain,XTrain_clip,YTrain_clip,LTrain_clip,XQuery,LQuery,LLQuery,XQuery_clip] = load_dataset(train_param);
%%
train_param.alpha=10;
train_param.lamda =1e3;
train_param.beta =1e3;
train_param.thetaP =1e1;
train_param.thetaW =1e1;
train_param.thetaU = 1e1;
train_param.kesi = 1e-3;
train_param.sigma=1e-3;
train_param.c=	40;
train_param.iter = 5;
train_param.current_bits=16;

[eva,~]=evaluate_WOCH(XTrain_clip,YTrain,XTrain_clip,LTrain,XQuery_clip,LQuery,train_param);

s/i_',num2str(val(i)),' j_',num2str(val(j)),' k_',num2str(val(k)),' m_',num2str(val(m)),'.mat'],'res');
            end
        end
    end
end
    
%[eva,t]=evaluate_WOCH(XTrain_clip,YTrain,XTrain_clip,LTrain,XQuery_clip,LQuery,train_param);
