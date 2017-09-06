
tic;
addpath('.\measure');
addpath('.\data');
data_sets={'emotions','CAL500','enron'};
record=zeros(10,11);

for didx=1:length(data_sets)
    data_set=data_sets(didx);
    data_set=data_set{1};
    load([data_set,'.mat']);
    fprintf([data_set,'\n']);
    data=normalize(data);
    for i=1:10 % 10-fold test
        fprintf('%d,',i);
        perm=randperms(i,:);
        n=length(perm);
        test_idx=perm(1:round(0.3*n)); %  70%/30% for train/test
        train_idx=1:size(data,1);
        train_idx(test_idx)=[];

        train_data=data(train_idx,:);
        train_targets=targets(train_idx,:);

        test_data=data(test_idx,:);
        test_targets=targets(test_idx,:);

%limo main
%default: step_size=0.01, epoch=10, lambda2=1, lambda1=1
        W=limo_main(train_data,train_targets,0.01,10,1,1);

%different threshold strategies for optimizing different classification measures

    %for ranking based performance measures, and thresholding for hamming loss 
        threshold=limo_row_threshold(train_data*W,train_data,train_targets,'hloss');
        [predict_labels,value]=limo_predict(W,test_data,threshold,'row_svm');
        measure=perform_measure(predict_labels,value,test_targets,...
                {'hamming loss','average precision','ranking loss',...
                 'coverage','one-error','micro-AUC','macro-AUC','instance-AUC'});

    %for macro-F1, we need another thresholding
        threshold_macro=limo_column_threshold(train_data*W,train_targets,'macro-F1');
        [predict_labels,value]=limo_predict(W,test_data,threshold_macro,'column');
        temp=perform_measure(predict_labels,value,test_targets,{'macro-F1'});
        measure.macro_F1=temp.macro_F1;
        
    %for micro-F1 and instance-F1, we need another thresholding
        threshold_micro=limo_column_threshold(train_data*W,train_targets,'micro-F1');
        [predict_labels,value]=limo_predict(W,test_data,threshold_micro,'column');
        temp=perform_measure(predict_labels,value,test_targets,{'micro-F1','instance-F1'});
        measure.micro_F1=temp.micro_F1;
        measure.instance_F1=temp.instance_F1;
        
        x=measure;
        
        record(i,:)=[x.hamming_loss,x.ranking_loss,x.one_error,x.coverage,x.average_precision,...
                     x.micro_F1,x.macro_F1,x.instance_F1,...
                     x.micro_AUC,x.macro_AUC,x.instance_AUC];
        
  

        
    end
    record_mean=mean(record);
    fprintf('\nhl:%.4f rl:%.4f oe:%.4f cov:%.4f ap:%.4f micro-F1:%.4f macro-F1:%.4f instance-F1:%.4f micro-AUC:%.4f macro-AUC:%.4f instance-AUC:%.4f\n',record_mean);

    %fprintf('\nrl:%f\tmicrof1:%f\tmacrof1:%f\texample_f1:%f\thl:%f\tap:%f\toe:%f\tcov:%f\tmacro_AUC:%f\texample_AUC:%f\tmicro_AUC:%f\n',record_mean);
   
    record_std=std(record);
    record_summary=[record_mean;record_std];
    
end
toc;
