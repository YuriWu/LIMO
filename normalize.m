function n_data=normalize(data)

n_data=zeros(size(data));
[~,d]=size(data);
for i=1:d
    n_data(:,i)=data(:,i)/max(abs(data(:,i)));
end

end