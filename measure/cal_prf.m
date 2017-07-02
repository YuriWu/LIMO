function [ Precision,Recall,F1 ] = cal_prf(X,Y,f1_type)
%SLPRF Calculate the average micro Precision, recall and F1 measure
% $ Syntax $
%   - slprf( X,Y,varargin)

%
% $ Description $
%   - X: the classifier's output labels for sample i : Outputs(:,i)with predicted label=1 else
%   -1
%   - Y: the real labels, each column for one sample , +1 for label, else -1 
%   - 'type':   the type of P,R,F: 
%               'micro'--micro-averaging (results are computed based on global sums over all decisions) (default ='micro')
%               'macro'--macro-averaging (results are computed on a per-category basis, then averaged over categories)
%       Micro-averaged scores tend to be dominated by the most commonly used categories, 
%       while macro-averaged scores tend to be dominated by the performance in rarely used categories. 
%   
% $ History $
%   - Created by Xiangnan Kong, on Jan 7, 2008
%   - Yuri Wu, 2016/5/11
%% parse and verify input arguments

%% calculate average Precision
X(X>0) = 1;X(X<=0) = 0;
Y(Y>0) = 1;Y(Y<=0) = 0;
XandY = X&Y;
if strcmp(f1_type,'micro')
    Precision=sum(XandY(:))/sum(X(:));
    Recall=sum(XandY(:))/sum(Y(:));
    F1=2*Precision*Recall/(Precision+Recall);
    F1(isnan(F1))=0;
end
if strcmp(f1_type,'macro')
    p=sum(XandY,1)./sum(X,1);
    r=sum(XandY,1)./sum(Y,1);
    f=2*p.*r./(p+r);
    f(isnan(f))=0;
    Precision = mean(p);
    Recall = mean(r);
    F1 = mean(f);
end
if strcmp(f1_type,'example')
    p=sum(XandY,2)./sum(X,2);
    r=sum(XandY,2)./sum(Y,2);
    f=2*p.*r./(p+r);
    f(isnan(f))=0;
    Precision = mean(p);
    Recall = mean(r);
    F1 = mean(f);
end
