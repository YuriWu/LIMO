function [ micro_F1, macro_F1, instance_F1 ] = cal_F1(X,Y)
%Calculate the F-measure on different averaging strategy
%
% Description
%   number of instances is n, label dimension is m
%   - X: n*m, the classifier's output labels for sample i : Outputs(:,i)with predicted label=1 else
%   -1 or 0
%   - Y: n*m, the real labels, each column for one sample , +1 for label, else -1 
%       Micro-averaged scores tend to be dominated by the most commonly used categories, 
%       while macro-averaged scores tend to be dominated by the performance in rarely used categories. 
%   $ History $
%   - Created by Xiangnan Kong, on Jan 7, 2008
%   - Edited Yuri Wu, 2016/5/11
%% parse and verify input arguments

%% calculate average Precision
X(X>0) = 1;X(X<=0) = 0;
Y(Y>0) = 1;Y(Y<=0) = 0;
XandY = X&Y;

Precision=sum(XandY(:))/sum(X(:));
Recall=sum(XandY(:))/sum(Y(:));
micro_F1=2*Precision*Recall/(Precision+Recall);
if isnan(micro_F1)
    fprintf('micro F1 is NaN');
    micro_F1=0;
end

p=sum(XandY,1)./sum(X,1);
r=sum(XandY,1)./sum(Y,1);
f=2*p.*r./(p+r);
f(isnan(f))=0;
macro_F1 = mean(f);

p=sum(XandY,2)./sum(X,2);
r=sum(XandY,2)./sum(Y,2);
f=2*p.*r./(p+r);
f(isnan(f))=0;
instance_F1 = mean(f);

end
