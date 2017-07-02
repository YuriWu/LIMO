function [threshold,min_hloss]=best_hloss_threshold(value,groundtruth)

groundtruth(groundtruth==-1)=0;
block=[value,groundtruth];
v=sortrows(block,1);
n=length(groundtruth);

all_positive=sum(groundtruth==1);
tp=all_positive;
fp=n-all_positive;
fn=0;

min_hloss=compute_hloss(fp,fn,n);
threshold=v(1,1)-1e-5;
for i=2:n
    
    gt_label=v(i-1,2);
    if gt_label==1
        tp=tp-1;
        fn=fn+1;
    else
        fp=fp-1;
        current_hloss=compute_hloss(fp,fn,n);
        if v(i,1)-v(i-1,1)<=1e-8 %to small difference
            continue;
        end
        if current_hloss<min_hloss
            min_hloss=current_hloss;
            threshold=( v(i,1)+v(i-1,1) )/2;
        end
    end
end


end

function hloss=compute_hloss(fp,fn,n)
    hloss= (fp+fn)/n;
end