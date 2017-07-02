function model=limo_row_threshold(value,train_data,train_targets,metric)

[n,~]=size(train_targets);
threshold=zeros(n,1);
if strcmp(metric,'example-F1')
    for i=1:n
        threshold(i,1)=best_f1_threshold(value(i,:)',train_targets(i,:)');
    end
elseif strcmp(metric,'hloss')
    for i=1:n
        threshold(i,1)=best_hloss_threshold(value(i,:)',train_targets(i,:)');
    end
else
    fprintf('unknown metric!');
end
model=svmtrain(threshold,train_data,'-s 3 -t 2 -c 1 -q'); %libsvm
%w=lscov(train_data,threshold);

end