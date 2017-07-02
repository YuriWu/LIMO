function threshold=limo_column_threshold(value,train_targets,metric)

labels=size(train_targets,2);
threshold=zeros(1,labels);
if strcmp(metric,'macro-F1')
    for i=1:labels
        threshold(1,i)=best_f1_threshold(value(:,i),train_targets(:,i));
    end
elseif strcmp(metric,'micro-F1') %global threshold for all labels
    v=value(:);
    tt=train_targets(:);
    for i=1:labels
        threshold(1,i)=best_f1_threshold(v,tt);
    end
elseif strcmp(metric,'hloss')
    for i=1:labels
        threshold(1,i)=best_hloss_threshold(value(:,i),train_targets(:,i));
    end
else
    fprintf('unknown metric!');
end

end