% Author:       Clum, Mixon, Villar, Xie.
% Filename:     sketch_and_solve_lower_bound.m
% Last edited:  20 November 2022 
% Description:  This function computes high-confidence Hoeffding Monte 
%               Carlo k-means lower bound and Markov Monte Carlo k-means 
%               lower bound descirbed in [1].
%               This Requires CVX [2] and SDPNAL+0.3 [2].
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
%               -s:
%               Sketch size.
%
%               -l:
%               The number of repeated k-means++ initialization.
%
%               -epsilon:
%               Error rate.
%
%               -u:
%               Truncation parameter for B_H. Set u < 0 to run the 
%               deterministic kmeans++ initialization instead of truncating
%               the sketched SDP value.
%
%               -bound_type:
%               If bound_type = "Hoeffding", run algorithm 3 in [1]. If
%               bound_type = "Markov", run algorithm 4 in [1].
%
% Outputs: 
%
%               -B_H:
%               Hoeffding Monte Carlo k-means lower bound by
%               sketch-and-solve approach.
%
%               -B_M:
%               Markov Monte Carlo k-means lower bound by 
%               sketch-and-solve approach.
%
%               -Time_SDP:
%               Runtime for l repeated sketched SDP
%
%               -Num_B_truncations:
%               Number of truncations when computing B_H
%           
%               -Total_B_truncation:
%               Total amount of truncations when computing B_H
%               
% References:
% [1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches 
%       to k-means clustering by semidefinite programming.
% [2] M. Grant, S. Boyd, CVX: Matlab software for disciplined convex 
%       programming.
% [3] D. F. Sun, L. Q. Yang, K. C. Toh, Sdpnal+: A majorized semismooth 
%       newton-cg augmented lagrangian method for semidefinite programming 
%       with nonnegative constraints.
% -------------------------------------------------------------------------

function [B_H, B_M, Time_SDP, Num_B_truncations, Total_B_truncation] ...
     = sketch_and_solve_lower_bound(X, k, s, l, epsilon, u, bound_type)

tic;

[n, ~] = size(X);
% Set warm_start = 1, if using warm start for SDPNAL+. 
warm_start = 1;
% Number of kmeans++ on each sketched data for warm start.
num_kmeans_subset = 50;
% SDPNAL+ tolerance. 
tolerance = [1e-5, 1e-6];
% SDPNAL+ maximum iteration.
max_iterations = 20000;
% Reduce SDPNAL+ tolerance to the second tolerance if SDPNAL+ 
% takes less than min_iterations to converge.
min_iterations = 200; 
% Truncating the distance matrix by max_D.
max_D = 1e8;

if u < 0 
    [almost_sure_b, ~] = deterministic_kmeans_ini(X, k);
end

if bound_type == "Hoeffding"
    sample_method = true;
elseif bound_type == "Markov"
    sample_method = false;
else
    sample_method = false;
    disp('Wrong bound_type. Run Markov lower bound by default.' )
end

SDP_sketched_value_SDPNAL = zeros(l, 1); % Sketched SDP 
SDP_skethced_value_corrected = zeros(l, 1); % Sketched SDP after correcting
count_neg_SDP = 0; % Count negtive obj after correcting

check_dual_feasible_corrected = zeros(3, l); % Dual feasibility of 
% (1) S symmetric; (2) S non-negative eigenvalue; (3) Z non-negative 
% (while forcing the equality constrain hold).

kmeans_min_each_subset = zeros(l, 1); % Minimum kmeans++ results 
% for each subset (upper bound of sketched SDP).


for i = 1 : l

    % Sketch
    sample_ind = datasample(1:n, s, 'Replace', sample_method);
    Yi = X(sample_ind, :);
    
    % Run Kmeans++ on sketched data for warm start
    idx_all = zeros(s, num_kmeans_subset);
    k_means_results_subset = zeros(num_kmeans_subset, 1);
    for j = 1 : num_kmeans_subset
        [idx, ~, sumd] = kmeans(Yi, k);
        idx_all(:, j) = idx;
        k_means_results_subset(j) = sum(sumd) / s;
    end
    [k_means_min, min_j] = min(k_means_results_subset);
    kmeans_min_each_subset(i) = k_means_min;
    idx_min = idx_all(:, min_j);

    % Construction initialization matrix from kmeans++ for warm start
    Y_feas = zeros(s, s);
    for ii = 1 : k
       % Use == to get an indicator for the elements in cluster k
       ind_k = (idx_min==ii);
       num_k = sum(ind_k);
       Y_block = zeros(s, s);
       if num_k > 0
        Y_block = 1/num_k * ( double(ind_k)*double(ind_k).' );
       end
        Y_feas = Y_feas + Y_block;
    end
    
    % Solve sketched SDP 
    [Objective_pre_correct, S0_vec, At_mat, D_vec, b, ...
        dual_feasible_SDPNAL] = sketched_SDP_SDPNAL(Yi, k, tolerance, ...
        min_iterations, max_iterations, warm_start, Y_feas, max_D);
    SDP_sketched_value_SDPNAL(i) = Objective_pre_correct;
    
    
    % Correcting the numerical dual certificate
    [Objective_corrected, dual_feasible_corrected] = ...
        correct_to_dual_feasible(S0_vec, At_mat, D_vec, b);
    SDP_skethced_value_corrected(i) = max(Objective_corrected, 0);

    % Count negtive objective after correcting
    if Objective_corrected <= 0
        count_neg_SDP = count_neg_SDP + 1;
    end
     
    Three_dual_feasible = zeros(3,1);
    Three_dual_feasible(1) = dual_feasible_SDPNAL(1);
    Three_dual_feasible(2) = dual_feasible_SDPNAL(2);
    Three_dual_feasible(3) = dual_feasible_corrected;
    check_dual_feasible_corrected(:,i) = Three_dual_feasible;

end


% Calculate lower bound
if u < 0 
    % No truncation on sketched SDP value
    Avg_SDP_sketched = sum(SDP_skethced_value_corrected) / l;
    % Hoeffding Monte Carlo k-means lower bound by sketch-and-solve
    B_H = Avg_SDP_sketched - sqrt( - almost_sure_b^2 * ...
        log(epsilon) / (2*l) );
else
    % Truncating the sketched SDP value
    SDP_skecthed_value_truncated = min(SDP_skethced_value_corrected, u);
    % Number of truncations
    Num_B_truncations = sum(SDP_skethced_value_corrected >= u);
    % Total amount of truncations
    Total_B_truncation = sum( max(0, SDP_skethced_value_corrected - u) );
    
    Avg_SDP_sketched_truncated = sum(SDP_skecthed_value_truncated) / l;
    % Hoeffding Monte Carlo k-means lower bound by sketch-and-solve
    B_H =  Avg_SDP_sketched_truncated - sqrt( - u^2 * ...
        log(epsilon) / (2*l) );
end
% Markov Monte Carlo k-means lower bound by sketch-and-solve
eps = nthroot(epsilon, l);
B_M = eps * min(SDP_skethced_value_corrected);


% Display 

% Check if SDP is solved correctly
disp('Sanity Checks and Validation (sketch_and_solve_lower_bound):');
if count_neg_SDP > 0
    disp('Negative SDP result occurs (SDP may not solved correctly).')
    disp(['Number of negative SDP is: ' num2str(count_neg_SDP)])
end
% Check dual feasibility
disp(['Maximum dual infeasible norm: ' ...
    num2str(max(check_dual_feasible_corrected, [], 'all'))]);
% Check if SDP is tight
Check_lower_bound = kmeans_min_each_subset - SDP_skethced_value_corrected;
disp(['Tightest SDP lower bound among all trial: ' ...
    num2str(min(Check_lower_bound))]);

Time_SDP = toc;

end