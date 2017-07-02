function [hammingloss,oneerror,coverage,rankingloss,averageprecision,micro_F1,macro_F1,instance_F1]=cal_result(Pre_Labels,Outputs,test_targets)
%hl,oe,coverage,rl,ap,micro_f1,macro_f1,example_f1
Outputs=Outputs';
Pre_Labels=Pre_Labels';
Pre_Labels(Pre_Labels==0)=-1;
test_targets(test_targets==0)=-1;
test_targets=test_targets';
%cal_result calculate the hamming loss,one error,coverage,ranking loss,average precision,precision,recall,f1
%
%Outputs          - A QxM array, the output of the ith testing instance on the jth class is stored in Outputs(j,i)
%Pre_Labels       - A QxM array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%test_targets     - A QxM array, if the ith test bag belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%

hammingloss=Hamming_loss(Pre_Labels,test_targets);
oneerror=One_error(Outputs',test_targets');
coverage=Coverage(Outputs,test_targets);
rankingloss=Ranking_loss(Outputs,test_targets);
averageprecision=Average_precision(Outputs,test_targets);
[ micro_F1, macro_F1, instance_F1 ] = cal_F1(Pre_Labels',test_targets');

%[ example_AUC,example_AUC,micro_AUC ] = cal_AUC(labels,scores);