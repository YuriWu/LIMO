function [label,value]=limo_predict(W,test_data,threshold,threshold_type)
%%a predict function works both for global threshold,
%%  column threshold and dummy label threshold
value=test_data*W;
label=zeros(size(value));
if strcmp(threshold_type,'column') || strcmp(threshold_type,'global')
    for i=1:size(value,1)
        label(i,value(i,:)>=threshold)=1;
    end
elseif strcmp(threshold_type,'dummy')
    for i=1:size(value,1)
        label(i,value(i,:)>value(i,end))=1;
    end
    %eliminate dummy;
    label=label(:,1:end-1); 
    value=value(:,1:end-1);
elseif strcmp(threshold_type,'row')
    threshold_value=test_data*threshold;
    thresholded_value=bsxfun(@minus,value,threshold_value);
    label(thresholded_value>0)=1;
elseif strcmp(threshold_type,'row_svm')
    dummy=zeros(size(value,1),1);
    threshold_model=threshold;
    threshold_value=svmpredict(dummy,test_data,threshold_model,'-q');
    thresholded_value=bsxfun(@minus,value,threshold_value);
    label(thresholded_value>0)=1;
else
    fprintf('\nerr\n');
end

end
