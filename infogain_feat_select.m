classdef infogain_feat_select < handle
    % Supervised Feature Selection
    %
    
    properties
        %% init properties
        data
        labels
        
        vocabulary
        vocabulary_nostop

        % generated properties
        n_trn
        classnum

        data_tf
        labels_tf_trn
        labels_tf_tst
        data_tf_trn
        data_tf_tst
        
        labels_ig_trn
        labels_ig_tst
        data_ig
        data_ig_tf
        data_ig_tfidf
        data_ig_tfidf_norm

        data_ig_tfidf_norm_trn
        data_ig_tfidf_norm_tst
        
        
        
    end
    
    methods
        function o = infogain_feat_select(data_trn, labels_trn, data_tst, labels_tst)
            % merge to a single dataset. (each row is a sample, columns are features)
            o.n_trn = length(labels_trn); % keeping the train/test split using this variable
            o.data = [data_trn; data_tst];
            o.labels = [labels_trn; labels_tst];
            o.classnum = length(unique(labels_trn));
        end
        
        function trim_stopwords(o, vocabulary, stopwords)
            %% FIND THE INDICES of stopwords
            stopword_index = zeros(size(stopwords));
            for i=1:size(stopwords,1) % deklete all stopword
                j=1; stop = 0;
                while j<=size(vocabulary,1) && ~stop
                    if isequal(vocabulary{j},stopwords{i})
                        stopword_index(i) = j;
                        stop = 1;
                    else
                        j = j+1;
                    end
                end
            end
            
            
            %% FIND THE INDICES WE KEEP (nonstopwords)
            nonstopword_index = setdiff([1:size(o.data,2)],stopword_index(stopword_index>0));
            o.data = o.data(:,nonstopword_index);
            o.vocabulary_nostop = vocabulary(nonstopword_index);
            
        end
        
        function eval_data_term_freq(o)
            
            % tf = 'term frequency'
            % data_tf has each document representation normalized by the number of words in
            % the document
            o.data_tf = o.data.';
            sd = sum(o.data,2);
            for i=1:size(o.data_tf,2)
                o.data_tf(:,i) = o.data_tf(:,i)/sd(i);   %% norm each word by the size of document
            end
            o.data_tf = o.data_tf';
            
            % remove features with zero prob_t (they cause NaN)
            data_trn = o.data_tf(1:o.n_trn,:);
            [N,~] = size(data_trn);
            data_trn = data_trn>0;
            prob_t = full(sum(data_trn)/N);
            o.data_tf = o.data_tf(:, prob_t~=0);
            
        end
        
        function infogain = get_info_gain_vec(o)
            % works on the Xtf (not Xtfidf)
            %calculate_infogain
            Xtf_train = o.data_tf(1:o.n_trn,:); % training subset
            f = find(sum(Xtf_train)==0);
            [N,d] = size(Xtf_train);
            Xtf_train = Xtf_train>0;
            prob_t = full(sum(Xtf_train)/N);
            joint_prob = zeros(o.classnum,d);
            cond_prob = zeros(o.classnum,d);
            for i=1:o.classnum
                joint_prob(i,:) = sum(Xtf_train(o.labels(1:o.n_trn)==i,:))/N;
                cond_prob(i,:) = joint_prob(i,:)./prob_t;
            end
            
            cat_prob = zeros(o.classnum,1); % cat = categories
            for i=1:o.classnum
                cat_prob(i) = sum(o.labels(1:o.n_trn)==i)/N;
            end
            cat_entropy = -sum(cat_prob.*log2(cat_prob));
            cat_prob = repmat(cat_prob,1,d);
            
            prob_t_neg = 1-prob_t;
            joint_prob_neg = cat_prob-joint_prob;
            cond_prob_neg = zeros(o.classnum,d);
            for i=1:o.classnum
                cond_prob_neg(i,:) = joint_prob_neg(i,:)./prob_t_neg;
            end
            
            
            cond_prob(cond_prob==0) = 1; %this is for the entropy...
            preentropy = -cond_prob.*log2(cond_prob);
            entrop = sum(preentropy);
            entrop = prob_t.*entrop;
            
            cond_prob_neg(cond_prob_neg==0) = 1; %this is for the entropy...
            preentropy_neg = -cond_prob_neg.*log2(cond_prob_neg);
            entrop_neg = sum(preentropy_neg);
            entrop_neg = prob_t_neg.*entrop_neg;
            infogain = entrop_neg+entrop;
            infogain = cat_entropy-infogain;
        end
        
        function feat_select_by_infogain(o, top_m)
            %         function [Xtfidf_norm_train,Xtfidf_norm_test,train_labels,test_labels,Xtf,Xtfidf,Xtfidf_norm,X] = get_top_m_terms(m)
            
            %             load all_data X test_labels train_labels;
            %             load all_infogain_data infogain; %#ok<NASGU>
            
            train_set_size = o.n_trn;
            train_labels = o.labels(1:o.n_trn);
            test_labels = o.labels((o.n_trn+1):end);
            
            infogain = o.get_info_gain_vec();
            
            [~,sortinfo] = sort(infogain,'descend');
            X = o.data(:,sortinfo(1:top_m));
            sd = sum(X,2);
            keep = find(sd~=0);
            keep_train = keep(keep<=train_set_size);
            keep_test = keep(keep>train_set_size);
            train_labels = train_labels(keep_train);
            test_labels = test_labels(keep_test-train_set_size);
            train_set_size = size(train_labels,1); %updating train_set_size after removal of docs with no words
            X = X(sd~=0,:);
            sd = sd(sd~=0);
            
            Xtf = X';
            
            for i=1:size(Xtf,2)
                Xtf(:,i) = Xtf(:,i)/sd(i);
            end
            Xtf = Xtf';
            
            df = sum(X>0); %the frequency of each word in the document
            idf = log(size(X,1)*(1./df)); %inverse document frequency
            
            
            %% xtfidf = xtf*idf
            Xtfidf = Xtf;
            for i=1:size(Xtfidf,2)
                Xtfidf(:,i) = Xtfidf(:,i)*idf(i);
            end
            
            %% norm by L2
            Xtfidf_norm = Xtfidf';
            for i=1:size(Xtfidf_norm,2)
                Xtfidf_norm(:,i) = Xtfidf_norm(:,i)/norm(Xtfidf_norm(:,i));
            end
            Xtfidf_norm = Xtfidf_norm';
            
            %% separate back to train and test
            Xtfidf_norm_train  = Xtfidf_norm(1:train_set_size,:);
            Xtfidf_norm_test  = Xtfidf_norm(train_set_size+1:end,:);

            %% changing object
            o.labels_ig_trn = train_labels;
            o.labels_ig_tst = test_labels;
            o.data_ig = X;
            o.data_ig_tf = Xtf;
            o.data_ig_tfidf = Xtfidf;
            o.data_ig_tfidf_norm =  Xtfidf_norm;
            o.data_ig_tfidf_norm_trn = Xtfidf_norm_train;
            o.data_ig_tfidf_norm_tst = Xtfidf_norm_test;
        
        end
        
    end
    
end
