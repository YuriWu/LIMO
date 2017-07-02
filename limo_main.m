function W=limo_main(train_data, train_targets,step_size_initial,loop,lambda2,lambda1)

train_targets(train_targets==-1)=0;
[n,m]=size(train_targets);
d=size(train_data,2);
norm_up=5;

W=limo_init(m,d,norm_up);

AW=zeros(size(W));
Anum=0;

for i=1:loop
   
    [W,AW,Anum]=sgd_limo_train(train_data,train_targets,W,AW,Anum,step_size_initial,lambda2,lambda1);
   
end

W=AW/Anum;

end

