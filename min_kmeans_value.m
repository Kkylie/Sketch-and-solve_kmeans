% Author:       Clum, Mixon, Villar, Xie.
% Filename:     min_kmeans_value.m
% Last edited:  9 November 2022 
% Description:  This function computes the smallest (normalized) 
%               k-means value by running k-means++ algorithm [1][2] l times
%               on the dataset X.
% 
%
% Inputs: 
%               -X: 
%               A n x d data matrix, where d denotes the dimension of 
%               the data and n denotes the number of points.
%
%               -k:
%               The number of clusters.
% 
%               -l:
%               The number of repeated k-means++ algorithm.
%
% Outputs: 
%               -min_vi: 
%               The smallest (normalized) k-means value.
%
%               -Time_k_plus_plus:
%               Runtime to compute min_vi.
%
% References:
% [1] S. Lloyd, Least squares quantization in PCM.
% [2] S. Vassilvitskii, D. Arthur, k-means++: The advantages of careful
%       seeding.
% -------------------------------------------------------------------------

function [min_vi, Time_k_plus_plus] = min_kmeans_value(X, k, l)

tic;

[n, ~] = size(X);
k_means_values = zeros(l,1);

for i=1:l
    [~,~,sumd, ~] = kmeans(X, k);
    k_means_values(i) = sum(sumd) / n;
end
min_vi = min(k_means_values);

Time_k_plus_plus = toc;

end