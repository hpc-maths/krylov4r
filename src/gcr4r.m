function [x, varargout] = gcr4r(A, b, varargin)
%gcr4r   Generalized Conjugate Residual Method (GCR) For Research.
%   x = gcr4r(A,b) attempts to solve the system of linear equations A*x = b
%   for x. The N-by-N coefficient matrix A must be square and the right
%   hand side column vector b must have length N. This uses the unrestarted
%   method.
%
%   x = gcr4r(AFUN,b) accepts a function handle AFUN instead of the matrix
%   A. AFUN(x) accepts a vector input x and returns the matrix-vector
%   product A*x. In all of the following syntaxes, you can replace A by
%   AFUN.
%
%   x = gcr4r(A,b,'restart',RESTART) restarts the method every RESTART iterations.
%   If RESTART is [] or specified, then the unrestarted method is used, as above.
%
%   x = gcr4r(A,b,'tol',TOL) specifies the tolerance of the method. If
%   TOL is [] or not specified, then the default value, 1e-12, is used.
%
%   x = gcr4r(A,b,'maxit',MAXIT) specifies the maximum number of
%   iterations.
%
%   x = gcr4r(A,b,'left_prec',HL,'right_prec',HR) use left and right 
%   preconditioners. If HL or HR is [] or unspecified then the corresponding
%   preconditioner is not applied. If HL is a matrix, then HL\x is applied to
%   vectors x. If it is a function handle, then it is simply applied to x.
%
%   x = gcr4r(A,b,'weight',W) specifies the weight  
%   matrix defining the hermitian inner product used in the algorithm,
%   computed by y'*W*x. W must be hermitian positive definite. 
%   A function can also be passed, returning how W is applied to a vector.
%   If W is [] or not specified, then identity is used.
%
%   x = gcr4r(A,b,'defl',Y,Z) specifies the deflation spaces.
%
%   x = gcr4r(A,b,'guess',x0) specifies the first initial guess. 
%   If x0 is [] or not specified, then an all zero vector is used.
%
%   x = gcr4r(A,b,HL,HR,'res',OPT) specifies how the
%   residual norm is computed. The convergence cirterion also uses the 
%   same configuration to compute the norm of b for assessing the relative
%   residual norm RELRES.
%   OPT='l': HL is applied: RELRES=norm(HL\r)/norm(HL\b)
%   OPT='r': HR is applied: RELRES=norm(HR\r)/norm(HR\b)
%   OPT='w': the W-norm is used.
%   OPT='' : RELRES=norm(r)/norm(b) 
%   Option combinations are allowed.
%   The default value depends on the presence of HL, HR, W, so that the
%   residual corresponds to the one that is minimized by the algorithm.
%   Examples:
%   If HL is provided, or HL and HR, then RELRES=norm(HL\r)/norm(HL\b).
%   If HR only, then RELRES=norm(r)/norm(b).
%   If W is provided, then the W-norm is used.
%
%   [x,FLAG] = gcr4r(A,b,...) also returns a convergence FLAG:
%    0 GCR converged to the desired tolerance TOL within MAXIT iterations.
%    1 GCR iterated MAXIT times but did not converge.
%    2 preconditioner HL or HR was ill-conditioned.
%    3 a breakdown occurred.
%
%   [x,FLAG,RELRES] = gcr4r(A,b,...) also returns the relative residual
%   NORM(b-A*x)/NORM(b). If FLAG is 0, then RELRES <= TOL. Note that with
%   preconditioners the preconditioned relative residual may be used.
%   See argument 'res' for details.
%
%   [x,FLAG,RELRES,ITER] = gcr4r(A,b,...) also returns both the iteration
%   number at which x was computed: 0 <= ITER <= MAXIT.
%
%   [x,FLAG,RELRES,ITER,ABSRESVEC] = gcr4r(A,b,...) also returns a vector of
%   the absolute residual norms at each iteration.
%   See argument 'res' for detail on how ABSRESVEC is computed according
%   to the presence of preconditioners and weighted norm.
%
%   [x,FLAG,RELRES,ITER,ABSRESVEC,RELRESVEC] = gcr4r(A,b,...) also returns a 
%   vector of the relative residual norms at each iteration, used as 
%   convergence criterion.
%   See argument 'res' for detail on how RELRESVEC is computed according
%   to the presence of preconditioners and weighted norm.
%
%   [x,FLAG,RELRES,ITER,ABSRESVEC,RELRESVEC,XVEC] = gcr4r(A,b,...) also 
%   returns the successive approximate solutions.
%
%   Author: P. Matalon, N. Spillane

    %% Argument processing
    
    % Deflation spaces
    with_deflation = 0;
    Y = [];
    Z = [];

    i = 1;
    while i < length(varargin)
        if strcmp(varargin{i}, 'defl')
            Y = varargin{i+1};
            Z = varargin{i+2};
        end
        i = i+1;
    end

    if ~isempty(Y) && ~isempty(Z)
        with_deflation = 1;
    end

    if with_deflation

        if isa(A, 'function_handle')
            error('GCR: Deflation is not implemented if A is a function. A must be a matrix.');
        end
    
        n = size(A, 1);
        if size(A, 2) ~= n
            error('GCR: A must be square.');
        end
        if isempty(Y) || isempty(Z)
            error('GCR: None or both deflation spaces Y and Z must be set.');
        end
        if ~all(size(Y) == size(Z))
            error('GCR: The deflation spaces Y and Z must have the same sizes.');
        end
        if size(Y, 1) ~= n
            error('GCR: The deflation spaces Y and Z must have the same numbers of rows as A.');
        end
        if size(Y, 2) > n
            error('GCR: The deflation spaces Y and Z must have lower or equal number of columns than A.');
        end
        if size(b, 1) ~= n
            error('GCR: The right-hand side b must have the same number of rows than A.');
        end
    end

    %% Solving

    if ~with_deflation
        % If no deflation space, then we apply the regular GCR algorithm
        [x, varargout{1:nargout-1}] = wp_gcr4r(A, b, varargin{:});
    else
        % Initializations
        AZ = A*Z;
        YtAZ = Y'*AZ;
        solve_YtAZ = @(b) YtAZ\b;
        
        apply_PD = @(x) x - AZ*solve_YtAZ(Y'*x);
        apply_QD = @(x) x -  Z*solve_YtAZ(Y'*A*x);
        
        % Deflated system: PD*A*x = PD*b
        apply_PDA = @(x) apply_PD(A*x);
        PDb = apply_PD(b);
    
        % Solve deflated system with GCR
        [x, varargout{1:nargout-1}] = wp_gcr4r(apply_PDA, PDb, varargin{:});
    
        % x = QD*x + (I-QD)x
        x = apply_QD(x) + Z*solve_YtAZ(Y'*b);
    end
end




%% ---------------------------------- %%
%     Weighted Preconditioned GCR      %
%           (no deflation)             %
%  ----------------------------------  %
function [x, flag, relres, iter, absresvec, relresvec, xvec] = wp_gcr4r(A, b, varargin)

    %% Argument processing

    if (nargin < 2)
        error(message('GCR: Not enough input arguments.'));
    end
    
    n = size(b, 1);
    if ~isa(A, 'function_handle')
        if size(A, 1) ~= size(A, 2)
            error('GCR: A must be square.');
        elseif size(A, 1) ~= n
            error('GCR: The right-hand side b must have the same number of rows than A.');
        end
    end

    if isa(A, 'function_handle')
        apply_A = @(x) A(x);
    else
        apply_A = @(x) A*x;
    end

    restart = [];
    tol     = [];
    maxit   = [];

    HL = [];
    HR = [];

    W = [];
    x0 = [];

    L_prec_res    = 1;
    R_prec_res    = 0;
    weighted_res  = 1;
    orthog_algo   = 'gs'; % Gram-Schmidt
    orthog_steps  = 1;
    breakdown_tol = 1e-12;

    i = 1;
    while i < length(varargin)
        if strcmp(varargin{i}, 'restart')
            restart = varargin{i+1};
        elseif strcmp(varargin{i}, 'tol')
            tol = varargin{i+1};
        elseif strcmp(varargin{i}, 'maxit')
            maxit = varargin{i+1};
        elseif strcmp(varargin{i}, 'left_prec')
            HL = varargin{i+1};
        elseif strcmp(varargin{i}, 'right_prec')
            HR = varargin{i+1};
        elseif strcmp(varargin{i}, 'weight')
            W = varargin{i+1};
        elseif strcmp(varargin{i}, 'guess')
            x0 = varargin{i+1};
        elseif strcmp(varargin{i}, 'res')
            L_prec_res   = ~isempty(strfind(varargin{i+1}, 'l'));
            R_prec_res   = ~isempty(strfind(varargin{i+1}, 'r'));
            weighted_res = ~isempty(strfind(varargin{i+1}, 'w'));
        elseif strcmp(varargin{i}, 'defl')
            i = i+1;
        elseif strcmp(varargin{i}, 'orthog_algo')
            orthog_algo = varargin{i+1};
        elseif strcmp(varargin{i}, 'orthog_steps')
            orthog_steps = varargin{i+1};
        elseif strcmp(varargin{i}, 'bkdwn_tol')
            breakdown_tol = varargin{i+1};
        else
            error(['GCR: unknown option ' varargin{i}]);
        end
        i = i+2;
    end

    if isempty(tol)
        tol = 1e-12;
    end

    if isempty(maxit)
        if isempty(restart)
            maxit = n;
        else
            maxit = ceil(n/restart);
        end
    end
    if isempty(restart)
        maxouter = 1;
        restart = maxit;
    else
        maxouter = ceil(maxit/restart);
    end

    if isempty(HL)
        HL = @(x) x;
    end
    if ~isa(HL, 'function_handle')
        if ~all(size(HL) == size(A))
            error('GCR: The left preconditioner HL must have the same size as the matrix A.');
        end
    end

    if isempty(HR)
        HR = @(x) x;
    end
    if ~isa(HR, 'function_handle')
        if ~all(size(HR) == size(A))
            error('GCR: The right preconditioner HR must have the same size as the matrix A.');
        end
    end

    if isa(HL, 'function_handle')
        apply_HL = @(x) HL(x);
    else
        apply_HL = @(x) HL\x;
    end
    if isa(HR, 'function_handle')
        apply_HR = @(x) HR(x);
    else
        apply_HR = @(x) HR\x;
    end

    apply_H = @(x) apply_HR(apply_HL(x));

    if isempty(W)
        dot_W = @(x,y) y'*x;
        norm_W = @(x) norm(x);
    elseif isa(W, 'function_handle')
        dot_W = @(x,y) y'*W(x);
        norm_W = @(x) sqrt(x'*W(x));
    else
        dot_W = @(x,y) y'*W*x;
        norm_W = @(x) sqrt(x'*W*x);
    end

    if isempty(x0)
        x0 = zeros(n, 1);
    end

    % Output argument indexes
    ARGOUT_ABSRESVEC = 5;
    ARGOUT_RELRESVEC = 6;
    ARGOUT_XVEC      = 7;

    %% How to compute the norm of b
    if weighted_res
        apply_res_norm = @(x) norm_W(x);
    else
        apply_res_norm = @(x) norm(x);
    end

    if L_prec_res && ~R_prec_res
        apply_res_prec = apply_HL;
    elseif L_prec_res && R_prec_res
        apply_res_prec = apply_H;
    elseif R_prec_res
        apply_res_prec = apply_HR;
    else
        apply_res_prec = @(x) x;
    end

    compute_b_norm = @(x) apply_res_norm(apply_res_prec(x));

    %% Initializations

    norm_HL_b_W = compute_b_norm(b);

    x = x0;
    if norm(x) == 0
        r = b;
    else
        r = b - apply_A(x);
    end

    % Memory allocation for the successive residual norms
    if nargout >= ARGOUT_ABSRESVEC
        absresvec = zeros(maxit+1, 1); % absolute
    end
    if nargout >= ARGOUT_RELRESVEC
        relresvec = zeros(maxit+1, 1); % relative
    end

    % Memory allocation for the successive solutions
    if nargout >= ARGOUT_XVEC
        xvec = zeros(n, maxit+1);
    end

    % Memory allocation to hold vectors
    p     = zeros(n, restart); % p_i  (research direction)
    Ap    = zeros(n, restart); % A*p_i
    HL_Ap = zeros(n, restart); % HL*A*p_i

    %% Iterations

    FLAG_CONVERGENCE = 0;
    FLAG_DIVERGENCE  = 1;
    FLAG_PREC_ISSUE  = 2;
    FLAG_BREAKDOWN   = 3;

    iter = 0;
    flag = FLAG_DIVERGENCE;

    for outer=1:maxouter

        % Residual (except on the first iteration)
        if outer > 1
            r = b - apply_A(x);
        end

        % Apply preconditioners
        HL_r = apply_HL(r);
        z = apply_HR(HL_r);
        if ~all(isfinite(z))
            warning('GCR: issue detected after applying the preconditioner.');
            flag = FLAG_PREC_ISSUE;
            break;
        end

        % Compute residual norm and check convergence
        absres = residual_norm(r, HL_r, z, apply_HR, apply_res_norm, L_prec_res, R_prec_res);
        relres = absres/norm_HL_b_W;
        if nargout >= ARGOUT_ABSRESVEC
            absresvec((outer-1)*restart + 1) = absres;
        end
        if nargout >= ARGOUT_RELRESVEC
            relresvec((outer-1)*restart + 1) = relres;
        end
        if nargout >= ARGOUT_XVEC
            xvec(:, (outer-1)*restart + 1) = x;
        end

        if (relres < tol)
            flag = FLAG_CONVERGENCE;
            break;
        end

        p(:,1)     = z/norm(z); % normalization to reduce the effects of round-off
        Ap(:,1)    = apply_A(p(:,1));
        HL_Ap(:,1) = apply_HL(Ap(:,1));

        for i=1:restart
    
            iter = iter+1;
    
            % Step length in the research direction
            delta = dot_W(HL_r, HL_Ap(:,i));
            gamma = dot_W(HL_Ap(:,i), HL_Ap(:,i));
            alpha = delta / gamma;

            if abs(delta) < breakdown_tol
                warning(['GCR: breakdown occurred at iteration ' num2str(iter) ' with delta=' num2str(delta) ' (< ' num2str(breakdown_tol) ')']);
                flag = FLAG_BREAKDOWN;
                iter = iter-1;
                break;
            end
    
            % Update
            x = x + alpha *  p(:,i); % solution
            r = r - alpha * Ap(:,i); % residual

            % Apply preconditioners
            HL_r = apply_HL(r);
            z = apply_HR(HL_r);
            if ~all(isfinite(z))
                warning('GCR: issue detected after applying the preconditioner.');
                flag = FLAG_PREC_ISSUE;
                break;
            end
    
            % Compute residual norm and check convergence
            absres = residual_norm(r, HL_r, z, apply_HR, apply_res_norm, L_prec_res, R_prec_res);
            relres = absres/norm_HL_b_W;
            if nargout >= ARGOUT_ABSRESVEC
                absresvec((outer-1)*restart + i+1) = absres;
            end
            if nargout >= ARGOUT_RELRESVEC
                relresvec((outer-1)*restart + i+1) = relres;
            end
            if nargout >= ARGOUT_XVEC
                xvec(:, (outer-1)*restart + i+1) = x;
            end
    
            if (relres < tol)
                flag = FLAG_CONVERGENCE;
                break;
            end
            if i == restart
                break;
            end

            % Orthogonalization of z against the p_i's
            p(:,i+1) = z;
            for os=1:orthog_steps
                if strcmp(orthog_algo, 'gs') % Gram-Schmidt
                    HL_Az = apply_HL(apply_A(p(:,i+1)));
                    for j=1:i
                        beta = dot_W(HL_Az, HL_Ap(:,j)) / dot_W(HL_Ap(:,j), HL_Ap(:,j));
                        p(:,i+1) = p(:,i+1) - beta*p(:,j);
                    end
                elseif strcmp(orthog_algo, 'mgs') % Modified Gram-Schmidt
                    for j=1:i
                        beta = dot_W(apply_HL(apply_A(p(:,i+1))), HL_Ap(:,j)) / dot_W(HL_Ap(:,j), HL_Ap(:,j));
                        p(:,i+1) = p(:,i+1) - beta*p(:,j);
                    end
                end
                p(:,i+1) = p(:,i+1)/norm(p(:,i+1)); % normalization to reduce the effects of round-off
            end
            Ap(:,i+1)    = apply_A(p(:,i+1));
            HL_Ap(:,i+1) = apply_HL(Ap(:,i+1));
        end

        if flag ~= FLAG_DIVERGENCE
            break;
        end
    end
    % Remove unused space
    if nargout >= ARGOUT_ABSRESVEC
        absresvec = absresvec(1:iter+1);
    end
    if nargout >= ARGOUT_RELRESVEC
        relresvec = relresvec(1:iter+1);
    end
    if nargout >= ARGOUT_XVEC
        xvec = xvec(:, 1:iter+1);
    end

    if flag == FLAG_DIVERGENCE && tol > 0
        warning(['GCR: the given tolerance could not be reached (maxit=' num2str(maxit) ').']);
    end
end


function absres = residual_norm(r, HL_r, H_r, apply_HR, apply_res_norm, L_prec_res, R_prec_res)
    if L_prec_res && ~R_prec_res
        absres = apply_res_norm(HL_r);        % ||HL r||
    elseif L_prec_res && R_prec_res
        absres = apply_res_norm(H_r);         % ||H r||
    elseif R_prec_res
        absres = apply_res_norm(apply_HR(r)); % ||HR r||
    else
        absres = apply_res_norm(r);           % ||r||
    end
end