function [train_param,XTrain,LTrain,LLTrain,HLTrain,YTrain,XTrain_clip,YTrain_clip,LTrain_clip,XQuery,LQuery,LLQuery,XQuery_clip] = load_dataset(train_param)
   
   if strcmp(train_param.ds_name, 'MIRFlickr-clip')
        fprintf(['-------load dataset------', '\n']);
        load('./MirFlickr-clip/MIRFlickr_weakly_features_CLIP_ViT_B_32_PCA_label.mat');
        load('./MirFlickr-clip/weakly_text_Bow_17833_404.mat');
        load('./MirFlickr-clip/labels_17833_38.mat');
        load('./MirFlickr-clip/Image_17833_4096.mat');

        train_param.fine_label_size=38;
        train_param.coarse_label_size=0;
        train_param.image_feature_size=4096;
        expected_chunksize=train_param.chunk_size;

        X_deep=Image_deep;
        L=labels;
        Y=tags;
        X_clip=image_features;
        Y_clip=PCA_text_features;
        L_clip=label_features;
        if train_param.normalizeX==1
            X_deep = bsxfun(@minus, X_deep, mean(X_deep,1));  % first center at 0
            X_deep = normr(double(X_deep));  % then scale to unit length
            X_clip=bsxfun(@minus, X_clip, mean(X_clip,1));
            X_clip=normr(double(X_clip));
        end
        
        
        R = randperm(size(X_deep,1));
        R_tr= R(1:16833);
        R_te=R(16834:17833);
        queryInds = R_te;
        sampleInds = R_tr;

        train_param.nchunks = ceil(train_param.N/expected_chunksize);

        train_param.chunksize = cell(train_param.nchunks,1);
        train_param.test_chunksize = cell(train_param.nchunks,1);

        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);
        LLTrain = cell(train_param.nchunks,1);
        HLTrain=cell(train_param.nchunks,1);
        YTrain=cell(train_param.nchunks,1);
        XTrain_clip=cell(train_param.nchunks,1);
        YTrain_clip=cell(train_param.nchunks,1);
        LTrain_clip=cell(train_param.nchunks,1);

        XQuery = cell(train_param.nchunks,1);
        LQuery = cell(train_param.nchunks,1);
        LLQuery = cell(train_param.nchunks,1);
        XQuery_clip=cell(train_param.nchunks,1);

        for subi = 1:train_param.nchunks
            XTrain{subi,1} = X_deep(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            LTrain{subi,1} = L(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            HLTrain{subi,1}=[1 1];
            YTrain{subi,1} = Y(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            XTrain_clip{subi,1} = X_clip(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            YTrain_clip{subi,1} = Y_clip(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            LTrain_clip{subi,1} = L_clip(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

            XQuery{subi,1} = X_deep(queryInds, :);
            LQuery{subi,1} = L(queryInds, :);
            XQuery_clip{subi,1} = X_clip(queryInds, :);
            [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

        end
        LLTrain=LTrain;
        LLQuery=LQuery;
        
    elseif strcmp(train_param.ds_name, 'MIRFlickr')
        fprintf(['-------load dataset------', '\n']);
        load('.\data\MIRFlickr\MIRFlickr-data.mat');

        train_param.fine_label_size=38;
        train_param.coarse_label_size=0;
        train_param.image_feature_size=4096;
        expected_chunksize=train_param.chunk_size;

        
        X_tr=XTrain;
        X_te=XTest;
        L_tr=LTrain;
        L_te=LTest;
        if train_param.normalizeX==1
            X_tr = bsxfun(@minus, X_tr, mean(X_tr,1));  % first center at 0
            X_tr = normr(double(X_tr));  % then scale to unit length
            X_te = bsxfun(@minus, X_te, mean(X_te,1));  % first center at 0
            X_te = normr(double(X_te));  % then scale to unit length
        end
        Y=YTrain;
        Z=ZTrain;

        R_tr = randperm(size(X_tr,1));
        R_te= randperm(size(X_te,1));
        queryInds = R_te;
        sampleInds = R_tr;

        train_param.nchunks = ceil(train_param.N/expected_chunksize);

        train_param.chunksize = cell(train_param.nchunks,1);
        train_param.test_chunksize = cell(train_param.nchunks,1);

        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);
        LLTrain = cell(train_param.nchunks,1);
        HLTrain=cell(train_param.nchunks,1);
        YTrain=cell(train_param.nchunks,1);
        ZTrain=cell(train_param.nchunks,1);

        XQuery = cell(train_param.nchunks,1);
        LQuery = cell(train_param.nchunks,1);
        LLQuery = cell(train_param.nchunks,1);

        for subi = 1:train_param.nchunks
            XTrain{subi,1} = X_tr(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            LTrain{subi,1} = L_tr(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            HLTrain{subi,1}=[1 1];
            YTrain{subi,1} = Y(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            ZTrain{subi,1} = Z(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
            [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

            XQuery{subi,1} = X_te(queryInds, :);
            LQuery{subi,1} = L_te(queryInds, :);
            [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

        end
        LLTrain=LTrain;
        LLQuery=LQuery;
        
    end    

    clear X_deep Y  L X_clip Y_clip L_clip subi queryInds sampleInds R

    fprintf('-------load data finished-------\n');
end

