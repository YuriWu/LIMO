function [ macro_AUC,example_AUC,micro_AUC ] = cal_AUC(labels,scores)

%Yuri Wu, 2016-04-10

[n,m]=size(labels);
macro_AUC=0;
valid_labels=0;
for i=1:m
    if size(unique(labels(:,i)),1)==2
        %[~,~,~,temp]=perfcurve(labels(:,i),scores(:,i),1);
        temp=fast_AUC(labels(:,i),scores(:,i));
        macro_AUC=macro_AUC+temp;
        valid_labels=valid_labels+1;
    end
end
macro_AUC=macro_AUC/valid_labels;

example_AUC=0;
valid_instances=0;
for i=1:n
    if size(unique(labels(i,:)),2)==2
        %[~,~,~,temp]=perfcurve(labels(i,:),scores(i,:),1);
        temp=fast_AUC(labels(i,:),scores(i,:));
        example_AUC=example_AUC+temp;
        valid_instances=valid_instances+1;
    end
end
example_AUC=example_AUC/valid_instances;

micro_AUC=0;
%[~,~,~,micro_AUC]=perfcurve(labels(:),scores(:),1);
micro_AUC=fast_AUC(labels(:),scores(:));

end