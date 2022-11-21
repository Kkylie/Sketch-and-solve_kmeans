% Author:       Clum, Mixon, Villar, Xie.
% Filename:     main.m
% Last edited:  20 November 2022 
% Description:  Run sketch-and-solve algorithms [1] to produce high
%               confidence lower bound for kmeans problem on MNIST 
%               data [5], synthetic datasets drawn according to Gaussian
%               mixture model, cloud cover data [2] and intrusion detection
%               data[4]. For comparison, we compute lower bounds by running 
%               k-means++ initialization [7].
%               This procedure prints out numerical results in Table 1 and
%               Table 2 in [1].
%               This Requires CVX [3] and SDPNAL+0.3 [6].
% 
%
% Input: 
%
% Output: 
%
%               
% References:
% [1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches 
%       to k-means clustering by semidefinite programming.
% [2] P. Collard, Cloud data set.
% [3] M. Grant, S. Boyd, CVX: Matlab software for disciplined convex 
%       programming.
% [4] KDD Cup 1999 dataset.
% [5] Y. LeCun, C. Cortes, MNIST handwritten digit database.
% [6] D. F. Sun, L. Q. Yang, K. C. Toh, Sdpnal+: A majorized semismooth 
%       newton-cg augmented lagrangian method for semidefinite programming 
%       with nonnegative constraints.
% [7] S. Vassilvitskii, D. Arthur, k-means++: The advantages of careful
%       seeding.
% -------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DATA_SET_name = 'NORM-10'; 

switch DATA_SET_name
    case 'MNIST'
        images = loadMNISTImages('./data/train-images.idx3-ubyte');
        X = images'; 
    case 'NORM-10'
        number_of_points = 10000;
        dimension = 5;
        number_of_centers = 10; 
        side_length = 500;
        variance = 1.0;
        X = Generate_Gaussian_Mixture(side_length, number_of_points, ...
            number_of_centers, dimension, variance);
    case 'NORM-25'
        number_of_points = 10000;
        dimension = 15;
        number_of_centers = 25; 
        side_length = 500;
        variance = 1.0;
        X = Generate_Gaussian_Mixture(side_length, number_of_points, ...
            number_of_centers, dimension, variance);
    case 'CLOUD'
        X = load("./data/cloud.data");
    case 'INTRUSION'
        Intrusion = readtable("./data/kddcup.data_10_percent_corrected.csv");
        Intrusion = Intrusion(:, [1;5;6;8;9;10;11;13;14;15;16;17;18;19;...
            20;23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41]);
        Intrusion = table2array(Intrusion);
        X = Intrusion;
end

[n, ~] = size(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Select Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l = 1000; % Number of trials
k = 10; % Number of cluster
s = 300; % Sketch size
epsilon = 0.01; % Error rate

% Set bound_type = "Hoeffding" to get B_H
% Set bound_type = "Markov" to get B_M
bound_type = "Hoeffding";
%bound_type = "Markov";

% Number sketched SDP
num_SDP = l; 
% Number of k-means++ on the whole dataset 
num_kmeans = l;
% Number of k-means++ init. on the whole dataset for comparison
num_kmeans_init = l; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Run k-means++ on the full dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[min_vi, Time_k_plus_plus] = min_kmeans_value(X, k, num_kmeans);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: Run k-means++ initialization for comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u_init = min_vi;
eps_init = epsilon;
[avg_Li, L_H, L_M, Time_k_init, Num_L_truncations, Total_L_truncation]...
    = kmeans_plusplus_lower_bound(X, k, num_kmeans_init, u_init, eps_init);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5: Run sketch-and-solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u_SDP = min_vi;
epsilon_SDP = epsilon;
[B_H, B_M, Time_SDP, Num_B_truncations, Total_B_truncation] = ...
    sketch_and_solve_lower_bound(X, k, s, num_SDP, epsilon_SDP, ...
    u_SDP, bound_type);


% Display 
disp(['Dataset Name: ' DATA_SET_name]);
disp('Parameters:');
disp(['n=' num2str(n)]);
disp(['k=' num2str(k)]);
disp(['s=' num2str(s)]);
disp(['l=' num2str(l)]);
disp(['error rate:' num2str(epsilon)]);

disp('Results:');
disp(['kmeans++ minimum value (min vi):' num2str(min_vi)]);
disp(['kmeans++ init avg lower (avg Li):' num2str(avg_Li)]);
disp(['kmeans++ Hoeffding lower Bound wtih truncations (L_H):' ...
    num2str(L_H)]);
disp(['kmeans++ Markov lower Bound (L_M):' num2str(L_M)]);
if bound_type == "Hoeffding"
    disp(['Hoeffding lower Bound wtih truncations (B_H):' num2str(B_H)]);
elseif bound_type == "Markov"
    disp(['Markov lower Bound (B_M):' num2str(B_M)]);
else
    disp(['Markov lower Bound (B_M):' num2str(B_M)]);
end

disp('Time:');
disp(['Time to compute l repeated initializations of kmeans++ (T_init):'...
    num2str(Time_k_init)]);
disp(['Time to compute min vi (T_k++):' num2str(Time_k_plus_plus)]);
disp(['Time to compute l randomly sketched SDPs (T_SDP):' ...
    num2str(Time_SDP)]);

disp('Sanity Checks and Validation (main):');
disp(['Number of kmeans truncations:' num2str(Num_L_truncations)]);
disp(['Total amount of kmeans truncations:' num2str(Total_L_truncation)]);
disp(['Number of SDP truncations:' num2str(Num_B_truncations)]);
disp(['Total amount of of SDP truncations:' num2str(Total_B_truncation)]);


