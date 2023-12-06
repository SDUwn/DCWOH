function [eva,train_time_round] = evaluate_WOCH(XTrain,YTrain,ZTrain,LTrain,XQuery,LQuery,train_param)
    eva=zeros(1,train_param.nchunks);
    train_time_round=zeros(1,train_param.nchunks);

    %% initialization
    
    X=cell2mat(XTrain);
    R=randsample(5000,1000);
    anchorX=X(R,:);
    HTrain=[];
    
    for chunki=1:train_param.nchunks
        fprintf('--chunk---%3d\n',chunki);
        
        XTrain_t=XTrain{chunki,:};
        ZTrain_t=ZTrain{chunki,:};
        ZTrain_t=ZTrain_t ./ sum(ZTrain_t.^2,2).^0.5;
        YTrain_t=YTrain{chunki,:};
        XQuery_t=XQuery{chunki,:};
        LQuery_t=LQuery{chunki,:};

        R=YTrain_t'*YTrain_t;
        S=XTrain_t*XTrain_t';
        
        tic 
        if chunki==1          
            [DD,BB,W1]=train_DCWOH(XTrain_t',YTrain_t',ZTrain_t',train_param,anchorX');          
        else         
            [DD,BB,W1]=train_DCWOH1(XTrain_t',YTrain_t',ZTrain_t',DD,BB,train_param,anchorX');
        end
        
        XTrain_t=Kernelize(XTrain_t,anchorX);
        B=cell2mat(BB)';
        W=(XTrain_t'*XTrain_t+train_param.thetaW*eye(size(anchorX,1)))\(XTrain_t'*BB{1,chunki}');
        
        train_time_round(1,chunki)=toc;
        fprintf('traintime=%d\n',train_time_round(1,chunki));
        
        fprintf('test beginning\n');
        
        XQuery_t=Kernelize(XQuery_t,anchorX);
        
        HTrain_t=single(XTrain_t*W>0);
        HTest=single(XQuery_t*W>0);
        HTrain=[HTrain;HTrain_t];
        
        
        Lbase=cell2mat(LTrain(1:chunki,:));
        Aff = affinity([], [], Lbase, LQuery_t, train_param);
        
        train_param.metric = 'mAP';
        eva(1,chunki)  = evaluate(HTrain, HTest, train_param, Aff);
        
    end
    
end


