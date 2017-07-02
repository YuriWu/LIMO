function [W,AW,Anum]=sgd_limo_train(train_data,train_targets,W,AW,Anum,step_size_initial,lambda2,lambda1)

[n,~]=size(train_targets);
label_cardinality=sum(sum(train_targets))/n;
positive_cache=cell(n,1);
negative_cahce=cell(n,1);
for i=1:n
    positive_cache{i}=find(train_targets(i,:)==1);
    negative_cahce{i}=find(train_targets(i,:)==0);
end
for i=1:n*round(label_cardinality)
    %%label-wise margin
    if lambda1~=0
        idx=randi(n);
        x=train_data(idx,:)';
        idx_pos=positive_cache{idx};
        if length(idx_pos)>1
            random_pos_idx=randsample(idx_pos,1);
        else
            random_pos_idx=idx_pos;
        end
        fp=W(:,random_pos_idx)'*x;

        idx_neg=negative_cahce{idx};
        if isempty(idx_neg) % incase no negative label in this instance, just skip           

        else
            if length(idx_neg)>1
                random_neg_idx=randsample(idx_neg,1);
            else
                random_neg_idx=idx_neg;
            end

            fn=W(:,random_neg_idx)'*x;
            if fn>fp-1 %violate
                wp=W(:,random_pos_idx);
                wn=W(:,random_neg_idx);

                tmp1=wp-step_size_initial*(-lambda1*x+wp);
                tmp2=wn-step_size_initial*(lambda1*x+wn);

                W(:,random_pos_idx)=tmp1;
                W(:,random_neg_idx)=tmp2;
            end
        end
    end

    %%instance-wise margin
    if lambda2~=0
        idx=randi(n);
        x_pos=train_data(idx,:)';
        idx_pos=positive_cache{idx};
        if length(idx_pos)>1
            random_pos_idx=randsample(idx_pos,1);
        else
            random_pos_idx=idx_pos;
        end
        fp=W(:,random_pos_idx)'*x_pos;
        
        neg_instance=train_targets(:,random_pos_idx)==0;
        idx_neg_instance=find(neg_instance==1);
        w=W(:,random_pos_idx);
        if isempty(idx_neg_instance) % incase no negative instance exists
            w=w-step_size_initial*(lambda2*(-x_pos)+w); %update w, only for pos, 'cause no neg exists
            W(:,random_pos_idx)=w;
        else
            if length(idx_neg_instance)>1 % avoid the all-but-one-negative case
                random_neg_idx=randsample(idx_neg_instance,1);
            elseif length(idx_neg_instance)==1 % only one negative
                random_neg_idx=idx_neg_instance;
            end
            x_neg=train_data(random_neg_idx,:)';
            fn=w'*x_neg;
            if fn>fp-1
                w=w-step_size_initial*(lambda2*(x_neg-x_pos)+w); %update w, half for pos, half for neg
                W(:,random_pos_idx)=w;
            end
        end  
    end

    AW=AW+W;
    Anum=Anum+1;
    
end
 
end