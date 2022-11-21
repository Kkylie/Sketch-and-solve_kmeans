% Author:       Clum, Mixon, Villar, Xie.
% Filename:     kmeans_plusplus_lower_bound.m
% Last edited:  9 November 2022 
% Description:  This function computes three (normalized) kmeans lower 
%               bound for comparison in [1] by running k-means++ 
%               initialization [2] l times on the dataset X.
% 
%
% Inputs: 
%               -X: 
%               A n x d data matrix, where d denotes the dimension of 
%               the data and N denotes the number of points.
%
%               -k:
%               The number of clusters.
% 
%               -l:
%               The number of repeated k-means++ initialization.
%
%               -u:
%               Truncation parameter for L_H.
%
%               -epsilon:
%               Error rate.
%
% Outputs: 
%               -avg_Li: 
%               Average Li (lower by kmeans++ bound without a probability
%               guarantee).
%
%               -L_H:
%               High confidence Hoeffding lower bound by kmeans++.
%
%               -L_M:
%               High confidence Markov lower bound by kmeans++.
%
%               -Time_k_init:
%               Runtime for l repeated initilaization of k-means++
%
%               -Num_L_truncations:
%               Number of truncations when computing L_H
%           
%               -Total_L_truncation:
%               Total amount of truncations when computing L_H
%               
% References:
% [1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches 
%       to k-means clustering by semidefinite programming.
% [2] S. Vassilvitskii, D. Arthur, k-means++: The advantages of careful
%       seeding.
% -------------------------------------------------------------------------

function [avg_Li, L_H, L_M, Time_k_init, Num_L_truncations, ...
    Total_L_truncation] = kmeans_plusplus_lower_bound(X, k, l, u, epsilon)

tic;

[n, ~] = size(X);
V0 = zeros(l,1);

for i=1:l
    % Run kmeans with one iteration as a lower bound of initialization
    % kmeans value V0
    [~,~,sumd, ~] = kmeans(X, k, 'MaxIter', 1); 
    V0(i) = sum(sumd) / n;
end

% Compute three kmeans++ lower bound
L = V0 / (8 * (log(k)+2) );
Li_truncated = min(L,u); % Truncation
Avg_Li_truncated = sum(Li_truncated) / l;
% Number of truncations
Num_L_truncations = sum(L >= u); 
% Total amount of truncations
Total_L_truncation = sum( max(0, L - u) );
% Average Li
avg_Li = sum(L) / l;
% Hoeffding lower bound by kmeans++
L_H = Avg_Li_truncated - sqrt( - u^2 * log(epsilon) / (2*l) );
% Markov lower bound by kmeans++
eps_kmeans = nthroot(epsilon, l);
L_M = eps_kmeans * min(L);

Time_k_init = toc;

end