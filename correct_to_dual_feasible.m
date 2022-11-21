% Author:       Clum, Mixon, Villar, Xie.
% Filename:     correct_to_dual_feasible.m
% Last edited:  14 November 2022 
% Description:  Correcting the numerical dual certificate of Peng-Wei SDP
%               [3] by fixing S in the dual program using CVX [2] as 
%               discussed in [1].
%
% Inputs: 
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
% Outputs: 
%               -objective_d: 
%               Normalized dual objective after correcting to dual
%               feasible point.
%
%               -check_dual_feasible:
%               Dual feasibility of P non-negative (while forcing the 
%               equality constrain hold).
%
% References:
% [1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches 
%       to k-means clustering by semidefinite programming.
% [2] M. Grant, S. Boyd, CVX: Matlab software for disciplined convex 
%       programming.
% [3] J. Peng, Y. Wei, Approximating k-means-type clustering via 
%       semidefinite programming.
% -------------------------------------------------------------------------

function [objective_d, check_dual_feasible] = ...
    correct_to_dual_feasible(S0_vec, At_mat, D_vec, b)

m = length(b);
n = m - 1;
R_vec = D_vec - S0_vec;

cvx_begin
   variable y_til(m,1)
   maximize dot(b,y_til)
   subject to 
        At_mat * y_til - R_vec <= 0
cvx_end

y = y_til;
objective_d = 1/(2*n) * dot(y, b);

% Check that the dual solution is feasible
A_adj_y = At_mat * y;
P_vec_alt = D_vec - A_adj_y - S0_vec;
check_dual_feasible = abs( min( min(P_vec_alt), 0) );
end
