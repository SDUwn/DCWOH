function [DD,BB,W]=train_WOCH0_v14(XTrain_t,YTrain_t,ZTrain_t,param,anchor)

    %% set the parameters
    nbits = param.current_bits;
    beta = param.beta;
%     alpha1 = param.alpha1;
%     alpha2 = param.alpha2;
    thetaP = param.thetaP;
    thetaW = param.thetaW;
    lamda=param.lamda;
    kesi=param.kesi;
    sigma=param.sigma;
    c=param.c;

    %% get the dimensions of features
    n= size(XTrain_t,2);
    dX=size(anchor,2);
    %dX = size(XTrain_t,1);
    dY = size(YTrain_t,1);


    %% initialization
    B = sign(randn(nbits, n));
    W = randn(dX, nbits);
    %U = randn(dY, dY);
    P = randn(nbits,c);
    F1 = randn(dX, c);
    F2 = randn(dY, c);
    V=randn(nbits,n);
    
    a=ones(n/c,1);
    r=randperm(c);
    b=eye(c);
    A=zeros(c,c);
    for i = 1:c
        A(i,:)=b(r(i),:);
    end
    G=kron(a,A);
    
    alpha1=0.5;
    alpha2=0.5;

    XTrain_t=Kernelize(XTrain_t',anchor')';
    
    %% iterative optimization
        for iter = 1:param.iter
        
            % update F
                F1=((XTrain_t*G))*pinv(G'*G);
                F2=((YTrain_t*G))*pinv(G'*G);
            
            % update G
            for i = 1:n
                obj = zeros(c, 1);
                tmp=eye(c);
                for j=1:c
                    obj(j,1) = obj(j,1) + alpha1*norm(XTrain_t(:,i)-F1(:,j))^2 + alpha2*norm(YTrain_t(:,i)-F2(:,j))^2+beta*norm(P(:,j)-V(:,i))^2;
                end
                [~, min_idx] = min(obj);
                G(i,:)=tmp(min_idx,:);
            end
            
            %update V
            Z=kesi*nbits*B*(ZTrain_t'*ZTrain_t)+sigma*B+beta*P*G';
            Temp = Z*Z'-(1/n)*Z*ones(n,1)*ones(1,n)*Z';
            [~,Lmd,OO] = svd(Temp); clear Temp
            idx = (diag(Lmd)>1e-6);
            O = OO(:,idx); 
            O_ = orth(OO(:,~idx));
            N = Z'*O/(sqrt(Lmd(idx,idx)))-(1/n)*ones(n,1)*(ones(1,n)*Z')*O/(sqrt(Lmd(idx,idx)));
            N_ = orth(randn(n,nbits-length(find(idx==1))));
            V = sqrt(n)*[O O_]*[N N_]';
            
            % update B
            Q = kesi*nbits*V*(ZTrain_t'*ZTrain_t)+sigma*V+lamda*W'*XTrain_t;
            B=sign(Q);
            
            %update P
            P=(V*G)/(G'*G+thetaP*eye(c));
            
            %%update W
            W=(XTrain_t*XTrain_t'+thetaW*eye(dX))\(XTrain_t*B');
            
            %update alpha
            alpha1=(sqrt(sum(sum((XTrain_t-F1*G').^2)))) / (sqrt(sum(sum((XTrain_t-F1*G').^2)))+sqrt(sum(sum((YTrain_t-F2*G').^2))));
            alpha2=(sqrt(sum(sum((YTrain_t-F2*G').^2)))) / (sqrt(sum(sum((XTrain_t-F1*G').^2)))+sqrt(sum(sum((YTrain_t-F2*G').^2))));
            fprintf('alpha1=%d  ',alpha1);fprintf('alpha2=%d\n',alpha2);
        end
        DD{1,1} = XTrain_t*G;
        DD{2,1} = YTrain_t*G;
        DD{1,2} = XTrain_t*B';
        DD{2,2} = YTrain_t*B';
        DD{1,3} = XTrain_t*XTrain_t';
        DD{2,3} = YTrain_t*YTrain_t';
        DD{1,4}= B*G;
        DD{1,5} = G'*G;
        DD{1,6} = V*ZTrain_t';
        DD{1,7} = B*ZTrain_t';
        DD{1,8} = V*G;
        DD{1,9} = B*G;
        %%
        BB{1,1}=B;
end
