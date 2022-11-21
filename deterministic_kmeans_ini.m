% Author:       Clum, Mixon, Villar, Xie.
% Filename:     deterministic_kmeans_ini.m
% Last edited:  9 November 2022 
% Description:  This function run the deterministic k-means++ 
%               initialization (Algorithm 2) in [1], and compute b in
%               Algorithm 3 in [1].
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
% Outnputs: 
%               -almost_sure_b: 
%               An almost sure upper bound for sketched SDP defined in
%               Algorithm 3 in [1].
%
%               -ind:
%               Indices of k well-separaed points that is selected by
%               deterministic k-means++ initialization.
%
% References:
% [1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches 
%       to k-means clustering by semidefinite programming.
% -------------------------------------------------------------------------

function [almost_sure_b, ind] = deterministic_kmeans_ini(X,k)

[n, ~] = size(X); 
ind = zeros(k,1);
ind(1) = 1;
min_dis = vecnorm( X - (ones(n,1) * X(1,:)), 2, 2);
for i = 2 : k
    [~, ind(i)] = max(min_dis);
    dis_new = vecnorm( X - (ones(n,1) * X(ind(i),:) ), 2, 2);
    min_dis = min(min_dis, dis_new);
end
almost_sure_b = ( max(min_dis) )^2;

end