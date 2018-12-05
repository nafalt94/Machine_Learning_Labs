clear;
load('Train_raw_data.mat');
load('Validation_raw_data.mat');
load('Test_raw_data.mat');
load('Labels.mat');
%%
%soundsc(Train_raw_data(:, 14000), 16000);
speechSpectrograms.m 