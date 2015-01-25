classdef supervised_feat_select < handle
    % Supervised Feature Selection
    %   
    
    properties
        data
        labels
        n_trn
        
        data_tf
        data_info_gain
        
    end
    
    methods
        function o = supervised_feat_select(data_trn, labels_trn, data_tst, labels_tst)
            if nargin > 2
                % if data is split to train/test then merge to a single
                % dataset. (each row is a sample, columns are features)
                o.n_trn = length(labels_trn);
                o.data = [data_trn; data_tst];
                o.labels = [labels_trn; labels_tst];
            else 
                o.data = data; % each row is a sample, columns are features
                o.labels = labels;
            end
        end
        
        function trim_stop_words(o, stopwords)
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
            nonstopword_index = setdiff([1:size(X,2)],stopword_index(stopword_index>0));
            X = X(:,nonstopword_index);
            vocabulary_nostop = vocabulary(nonstopword_index);
            
        end
        
    end
    
end

