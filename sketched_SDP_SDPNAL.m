% Author:       Clum, Mixon, Villar, Xie.
% Filename:     sketched_SDP_SDPNAL.m
% Last edited:  14 November 2022 
% Description:  This function uses SDPNAL+ [3] to solve Peng-Wei SDP [2]  
%               with some modifications discussed in [1].
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
%               -tolerance:
%               A decreasing sequence of SDPNAL+ tolerance. Reduce to the
%               next tolerance if SDPNAL+ converges with fewer iteration
%               than min_iterations.
%
%               -min_iterations:
%               Reduce SDPNAL+ tolerance to the next tolerance if SDPNAL+ 
%               takes less than min_iterations to converge.
%
%               -max_iterations:
%               SDPNAL+ maximum iteration.
%   
%               -warm_start:
%               If warm_start = 1, run SDPNAL+ with X_feas as
%               initialization. Otherwise, run SDPNAL+ without warm start.
%
%               -X_feas:
%               Initialized primal variable for SDP.
%
%               -max_D:
%               Truncating the distance matrix by max_D.
%
% Outputs: 
%               -objective_d: 
%               Normalized dual objective solved by SDPNAL+.
%               
%               -S0_vec:
%               Vectorized dual variable S0 solved by SDPNAL+.
%
%               -At_mat:
%               The linear equality constrants in dual program, A adjoint.
%
%               -D_vec:
%               Vectorized pairwise distance matrix.
%
%               -b:
%               The b vector in the dual objective function.
%
%               -check_dual_feasible:
%               Dual feasibility of (1) S0 symmetric; (2) S0 non-negative 
%               eigenvalue; (3) P0 symmetric; (4) P0 non-negative; (5) the
%               equality constrain; (6) P0 non-negative (while forcing 
%               the equality constrain hold).
%
% References:
% [1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches 
%       to k-means clustering by semidefinite programming.
% [2] J. Peng, Y. Wei, Approximating k-means-type clustering via 
%       semidefinite programming.
% [3] D. F. Sun, L. Q. Yang, K. C. Toh, Sdpnal+: A majorized semismooth 
%       newton-cg augmented lagrangian method for semidefinite programming 
%       with nonnegative constraints.
% -------------------------------------------------------------------------

function [objective_d, S0_vec, At_mat, D_vec, b, check_dual_feasible] = ...
    sketched_SDP_SDPNAL(X, k, tolerance, min_iterations,...
    max_iterations, warm_start, X_feas, max_D)

n = size(X, 1);

% Construction of distance squared matrix
D = zeros(n, n);
for ii = 1 : n
    for jj = 1 : n
        D(ii, jj) = norm( X(ii, :) - X(jj, :) )^2;
    end
end

% Truncating the distance matrix
D = min(D, max_D * ones(n,n));

% SDP definition for SDPNAL+
C{1} = D;
blk{1,1} = 's'; blk{1,2} = n;
b = zeros(n+1, 1);
Auxt = spalloc(n*(n+1)/2, n+1, 5*n);
Auxt(:,1) = svec(blk(1,:), eye(n),1);
b(1,1) = k;
idx = 2;
for i = 1 : n
    A = zeros(n, n);
    A(:, i) = ones(n, 1);
    A(i, :) = A(i, :) + ones(1, n);
    b(idx, 1) = 2;
    Auxt(:, idx) = svec(blk(1,:), A, 1);
    idx = idx + 1;
end
At{1} = sparse(Auxt);

% SDPNAL+ solver setting
num_tol = length(tolerance);
OPTIONS.maxiter = max_iterations; 
OPTIONS.tol = tolerance(1); 
OPTIONS.printlevel = 2;
OPTIONS.stopoption = 0; % prevent the solver from stopping prematurely

% Run SDPNAL+
if warm_start
    X_initial{1} = X_feas;
    [obj, Z, s_sdpnal, y0, S0, P0, y2, v, info, runhist] = ...
        sdpnalplus(blk, At, C, b, 0, [], [], [], [], OPTIONS, X_initial);
else
    [obj, Z, s_sdpnal, y0, S0, P0, y2, v, info, runhist] = ...
        sdpnalplus(blk, At, C, b, 0, [], [], [], [], OPTIONS);
end

% Reduce tolerance
if num_tol > 1
    for i = 2 : num_tol
        num_iter = info.iter;
        if num_iter < min_iterations
            OPTIONS.tol = tolerance(i);
            [obj, Z, s_sdpnal, y0, S0, P0, y2, v, info, runhist] = ...
                sdpnalplus(blk, At, C, b, 0, [], [], [], [], ...
                OPTIONS, Z, s_sdpnal, y0, S0, P0, y2, v);
        else
            break;
        end
    end
end

% Resulting normalized objective value
Z = cell2mat(Z);
objective_p = 1/(2*n) * trace(Z*D);
objective_d = 1/(2*n) * (b.'*y0);

% Check feasibility
At_mat = sparse(Auxt);
S0_mat = cell2mat(S0);
P0_mat = cell2mat(P0);
A_adj_y0 = At_mat * y0;
check_sym_S0 = norm(S0_mat - S0_mat');
check_sym_P0 = norm(P0_mat - P0_mat');
eval_S0 = eig(S0_mat);
check_psd_S0 = abs( min( min(eval_S0), 0) );
check_nneg_P0 = abs( min( min(P0_mat,[],'all'), 0) );
D_vec = svec(blk(1,:), D, 1);
S0_vec = svec(blk(1,:), S0_mat, 1);
P0_vec = svec(blk(1,:),P0_mat,1);
check_eq = norm(A_adj_y0 + S0_vec + P0_vec - D_vec);
P0_vec_alt = D_vec - A_adj_y0 - S0_vec;
check_nneg_P0_alt = abs( min( min(P0_vec_alt), 0) );

check_dual_feasible = zeros(1,6);
check_dual_feasible(1) = check_sym_S0;
check_dual_feasible(2) = check_psd_S0;
check_dual_feasible(3) = check_sym_P0;
check_dual_feasible(4) = check_nneg_P0;
check_dual_feasible(5) = check_eq;
check_dual_feasible(6) = check_nneg_P0_alt;

end
