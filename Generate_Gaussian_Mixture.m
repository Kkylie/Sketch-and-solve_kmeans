% Author:       Clum, Mixon, Villar, Xie.
% Filename:     Generate_Gaussian_Mixture.m
% Last edited:  20 November 2022 
% Description:  Generates data from a mixture of Gaussians model as in 
%               NORM-10 and NORM-25 of the paper kmeans++ [1].
% 
%
% Inputs: 
%               -side_length: 
%               Side length of the hypercube, where the center are drawn
%               uniformly from this hypercube. 
%
%               -number_of_points:
%               The number of data points.
% 
%               -number_of_centers:
%               The number of cluster centers.
%
%               -dimension:
%               The dimension of the data.
%
%               -variance:
%               Variance of the Gaussian distribution. 
%
% Outputs: 
%
%               -X:
%               A n x d data matrix generated from Gaussian mixture model,
%               where d denotes the dimension of the data and n denotes 
%               the number of points.
%               
% References:
% [1] S. Vassilvitskii, D. Arthur, k-means++: The advantages of careful
%       seeding.
% -------------------------------------------------------------------------

function X = Generate_Gaussian_Mixture(side_length, number_of_points, ...
    number_of_centers, dimension, variance)

% Dimension of ambient space
m = dimension;
% Number of gaussians
k = number_of_centers;
% Number of points
n = number_of_points;

% Choose the means uniformly from a hypercube with side length side_length
B = side_length;
centers = (rand(k,m) - 1/2*ones(k,m)) * B;

% Initialize n random points
X = zeros(n,m);
for i = 1 : n
   % Choose the gaussianfi
   index = randi([1, k],1,1);
   Mu(i,:) = centers(index,:);
   G(i,:) = normrnd(0,variance,1,m);
   X(i,:) = Mu(i,:) + G(i,:);
end

end

