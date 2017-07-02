function [threshold,max_f1]=best_f1_threshold(value,groundtruth)

groundtruth(groundtruth==-1)=0;
block=[value,groundtruth];
v=sortrows(block,1);
n=length(groundtruth);

all_positive=sum(groundtruth==1);
tp=all_positive;
fp=n-all_positive;
fn=0;
max_f1=compute_f1(tp,fp,fn);
threshold=v(1,1)-1e-5;
%view=zeros(1,n);
pos_idx=find(v(:,2)==1);
last_pos_idx=0;
for i=1:length(pos_idx)
    current_idx=pos_idx(i);
    fp=fp-(current_idx-last_pos_idx-1);
    current_f1=compute_f1(tp,fp,fn);
    if current_f1>max_f1
        max_f1=current_f1;
        threshold=( v(current_idx,1)+v(current_idx-1,1) )/2;
    end
    last_pos_idx=current_idx;
    tp=tp-1;
    fn=fn+1;
end
if isempty(pos_idx)
    threshold=v(end,1)+1e-5;
end
% for i=2:n
%     if v(i,2)==1 % threshold between the last 0 -> f1 peak 
%         current_f1=compute_f1(tp,fp,fn);
%         %view(1,i)=current_f1;
%         if current_f1>max_f1
%             max_f1=current_f1;
%             threshold=( v(i,1)+v(i-1,1) )/2;
%         end
%         tp=tp-1;
%         fn=fn+1;
%     else
%         fp=fp-1; % keep growing f1
% %         if v(i,1)-v(i-1,1)<=0.001 %to small difference
% %             continue;
% %         end      
%     end
% end


end

function f1=compute_f1(tp,fp,fn)
    if tp+fp>0
        precision=tp/(tp+fp);
    else
        f1=0;
        return;
    end
    if tp+fn>0
        recall=tp/(tp+fn);
    else
        f1=0;
        return;
    end
    if precision+recall==0
        f1=0;
        return;
    else
        f1=2*precision*recall/(precision+recall);
    end  
end