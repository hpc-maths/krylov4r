addpath('test_cases');

close all

%% System to solve
A = convdiff(50, 0.01); % Convection diffusion problem
b = ones(size(A, 1), 1);

%% Solver parameters
restart = [];
tol     = 1e-10;
maxit   = size(A, 1);

% Preconditioners
[L, U] = ilu(A);
HL = L;
HR = U;
%HL = [];
%HR = [];

norm_b = norm(b);
norm_HLb = norm(HL\b);
norm_HRb = norm(HR\b);
norm_Hb = norm(HR\(HL\b));

%% --------------- No preconditioner

%% GMRES
[~,~,~,~,resvec] = gmres(A, b, restart, tol, maxit);

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'o');
title(axes, 'No preconditioner');
ylabel(axes, '||H(b-Ax)||/||Hb||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,resvec] = gcr4r(A, b, 'restart', restart, 'tol', tol);

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES','GCR');

%% -------------- Left preconditioner

%% GMRES
[~,~,~,~,resvec] = gmres(A, b, restart, tol, maxit, HL, []);

figure; axes = gca;
semilogy(axes, 0:length(resvec)-1, resvec/norm_HLb, 'Marker', 'o');
title(axes, 'Left preconditioner');
ylabel(axes, '||HL(b-Ax)||/||HL b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,resvec] = gcr4r(A, b, 'left_prec', HL, 'restart', restart, 'tol', tol);

semilogy(axes, 0:length(resvec)-1, resvec/norm_HLb, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES (matlab)','GCR');


%% -------------- Right preconditioner

%% GMRES
[~,~,~,~,resvec] = gmres(A, b, restart, tol, maxit, [], HR);

figure; axes = gca;
semilogy(axes, 0:length(resvec)-1, resvec/norm_HRb, 'Marker', 'o');
title(axes, 'Right preconditioner');
ylabel(axes, '||HR(b-Ax)||/||HR b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,resvec] = gcr4r(A, b, 'right_prec', HR, 'restart', restart, 'tol', tol, 'res', 'r'); % Since gmres includes the right prec in the residual, we do it too

semilogy(axes, 0:length(resvec)-1, resvec/norm_HRb, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES (matlab)','GCR');


%% -------------- Left-right preconditioners

%% GMRES
[~,~,~,~,resvec] = gmres(A, b, restart, tol, maxit, HL, HR);

figure; axes = gca;
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'o');
title(axes, 'Left-right preconditioners');
ylabel(axes, '||H(b-Ax)||/||H b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,resvec] = gcr4r(A, b, 'left_prec', HL, 'right_prec', HR, 'restart', restart, 'tol', tol, 'res', 'lr'); % Since gmres includes the right prec in the residual, we do it too

semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES (matlab)','GCR');
