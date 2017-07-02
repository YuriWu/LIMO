function hamming_loss=measure_hamming_loss(predicted_labels,targets)

    [num_instance,num_class]=size(predicted_labels);
    missing_pairs=sum(sum(predicted_labels~=targets));
    hamming_loss=missing_pairs/(num_class*num_instance);
end