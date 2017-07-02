required_measures={'hamming loss','average precision','ranking loss',...
                    'coverage','one-error','micro-F1','macro-F1',...
                    'instance-F1','micro-AUC','macro-AUC','instance-AUC'};
load('sample.mat'); %predicted_labels,predicted_values,targets in it

x=perform_measure(predicted_labels,predicted_values,targets,required_measures);


cal_result(predicted_labels,predicted_values,targets);

