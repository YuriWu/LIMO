function W=limo_init(n_class,d,norm_up)

W=normrnd(0,1/sqrt(d),d,n_class); % D*n_class

for k=1:n_class
    tmp1=W(:,k);
    if(tmp1>norm_up)
        W(:,k)=tmp1*norm_up/norm(tmp1);
    end
end