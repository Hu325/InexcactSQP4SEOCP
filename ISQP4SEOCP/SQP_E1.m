function [L2_error_U] = SQP_E1(TOF, Nx, Ny)
%% ISQP in Neumann boundary condition case
%--------------------------------------------------------------------------
% Mesh information 
%--------------------------------------------------------------------------
    format long;
    g_1 = 0;
    g_2 = 1;
    pde.start_point_x=g_1;
    pde.end_point_x=g_2;
    pde.start_point_y=g_1;
    pde.end_point_y=g_2;
    [P,E,T] = meshGeneration(pde,Nx,Ny);              % spatial mesh
    cotNum = size(P, 2);                              % nodes number
%--------------------------------------------------------------------------
% Initialize the data 
%--------------------------------------------------------------------------
    Ud = zeros(cotNum, 1); 
    Yd = zeros(cotNum, 1); 
    Ybar = zeros(cotNum, 1); 
    Pd = zeros(cotNum, 1); 
    E_omega = zeros(cotNum, 1);
    UN = 0.3 * ones(cotNum,1);                  % initial control
    YN = 0.3 * ones(cotNum,1);                  % initial state
    PN = 0.3 * ones(cotNum,1);                  % initial adjoint
    B = massMatrix(P,T);                        % Aij = [φj,φi]
    GB = gradgradMatrix(P,T);                   % Aij = [▽φj,▽φi]
%--------------------------------------------------------------------------
% Function information 
%--------------------------------------------------------------------------  
    lambda = 1e-1;                              % regularization parameter
    for i = 1 : cotNum
         Ud(i) = u_opt(P(1, i), P(2, i));       % exact control
         Yd(i) = y_d(P(1, i), P(2, i));         % desired state
         Ybar(i) = y_bar(P(1, i), P(2, i));     % exact state
         Pd(i) = p_d(P(1, i), P(2, i));         % exact adjoint
         E_omega(i) = e_omega(P(1, i), P(2, i));% right-hand function
    end
    iter = 1;                                   % initial iteration count
%--------------------------------------------------------------------------
% SQP  
%--------------------------------------------------------------------------  
    SQPt = tic; 
    while 1
        [new_Un, new_Yn, new_Pn] = subQP(TOF, g_1, g_2, P, T, B, GB, UN, YN, PN, 0.15, 0.85, E_omega, lambda, Yd, iter); 
        NEW=[(new_Un'*B*new_Un + UN'*B*UN - 2*new_Un'*B*UN)^(1/2) (new_Yn'*B*new_Yn + YN'*B*YN - 2*new_Yn'*B*YN)^(1/2) (new_Pn'*B*new_Pn + PN'*B*PN - 2*new_Pn'*B*PN)^(1/2)];
        if   max(NEW) <= 1e-10 
            break;
        end  
        UN = new_Un;
        YN = new_Yn;
        PN = new_Pn;
        iter = iter+1; 
    end
    k = iter;
    SQPtime = toc(SQPt);
    disp(['SQP 迭代步数', num2str(k), ', SQP 累计用时', num2str(SQPtime), '秒.'])
%--------------------------------------------------------------------------
% Plot the figures
%--------------------------------------------------------------------------
    [r, c] = ndgrid(g_1 : (g_2/Nx) : g_2);
    matrix_U_N = reshape(new_Un, Nx + 1, Nx + 1)';
    u_n = griddedInterpolant(r, c, matrix_U_N, 'spline');
    x1 = g_1:g_2/10000:g_2;
    x2 = g_1:g_2/10000:g_2;
    [X1,X2] = ndgrid(x1,x2);
    F1 = (u_n(X1,X2)-u_opt(X1,X2)).^2;
    L2_error_U = sqrt(trapz(x2,trapz(x1,F1,2)));
     
    surfX = linspace(g_1, g_2, Nx+1);
    surfY = linspace(g_1, g_2, Ny+1);
    surfnew_Un = reshape(new_Un, Nx+1, Ny+1);
    surfUerr = reshape(new_Un - Ud, Nx+1, Ny+1);
    
    figure(1);
    s1 = surf(surfX, surfY, surfnew_Un, 'FaceAlpha', 1);
    s1.EdgeColor = 'none';
    axis([g_1 g_2 g_1 g_2]) 
    title('ISQP numerical solution')
    
    figure(2);
    s3 = surf(surfX, surfY, surfUerr, 'FaceAlpha',1);
    s3.EdgeColor = 'none';
    axis([g_1 g_2 g_1 g_2]) 
    title('Difference between numerical solution and exact solution')

end
%--------------------------------------------------------------------------
% Sub functions
%--------------------------------------------------------------------------
function u = u_opt(x, ~)
    u = min(max(3*x.^2 - 2*x.^3, 0.150), 0.85);
end
function y = y_bar(x, ~)
    y= -16*x.^4 + 32*x.^3 - 16*x.^2 + 1;
end
function e = e_omega(x, ~)
    e = 192*x.^2 - 192*x + 32 + y_bar(x, []) + y_bar(x, []).^3 - u_opt(x,[]); 
end
function p = p_d(x, ~)
    p = (2*x.^3 - 3*x.^2).*1e-1;
end
function y = y_d(x, ~)
    y = (12*x - 6).*1e-1 - p_d(x, []) - 3*y_bar(x, []).^2.*p_d(x, []) + y_bar(x, []);
end
%--------------------------------------------------------------------------
function [p,e,t]=meshGeneration(pde,Nx,Ny)
[p,e,t]=uniform_trimesh_delaunay(pde.start_point_x,pde.end_point_x,pde.start_point_y,pde.end_point_y,Nx,Ny);
end
function [p,e,t]=uniform_trimesh_delaunay(x_start,x_end,y_start,y_end,n_x,n_y)
h_x=(x_end-x_start)/n_x;
h_y=(y_end-y_start)/n_y;

[X,Y] = meshgrid(x_start:h_x:x_end,y_start:h_y:y_end);
length=(n_x+1)*(n_y+1);
x=reshape(X,length,1);
y=reshape(Y,length,1);
p=[x,y]';
t=delaunay(x,y)';
t=[t;ones(1,size(t,2))];
loc_1=find(p(1,:)==x_start);
loc_2=find(p(1,:)==x_end);
loc_3=find(p(2,:)==y_start);
loc_4=find(p(2,:)==y_end);
e=[loc_1(1:n_y),loc_2(1:n_y),loc_3(1:n_x),loc_4(1:n_x);loc_1(2:n_y+1),loc_2(2:n_y+1),loc_3(2:n_x+1),loc_4(2:n_x+1)];
end
function M = massMatrix(p,t)
%% Mass matrix: Aij = [φj,φi]
sDof=size(p,2);
nel=size(t,2);
nd=t(1:3,:);
a=p(:,nd(1,:));
b=p(:,nd(2,:));
c=p(:,nd(3,:));
x1=a(1,:); y1=a(2,:);
x2=b(1,:); y2=b(2,:);
x3=c(1,:); y3=c(2,:);
S=0.5.*(x2.*y3+x1.*y2+x3.*y1-x2.*y1-x1.*y3-x3.*y2);
ii=reshape([nd;nd;nd],1,9*nel);
jj=reshape([nd([1 2 3],:);nd([2 3 1],:);nd([3 1 2],:)],1,9*nel);
Mij=[repmat(S./6,3,1);repmat(S./12,6,1)]; 
M=sparse(ii,jj,Mij,sDof,sDof);
end
function A = gradgradMatrix(p,t)
%% Generate stiffness matrix: Aij = [▽φj,▽φi]
sDof=size(p,2);
nel=size(t,2);
nd=t(1:3,:);
a=p(:,nd(1,:));
b=p(:,nd(2,:));
c=p(:,nd(3,:));
x1=a(1,:); y1=a(2,:);
x2=b(1,:); y2=b(2,:);
x3=c(1,:); y3=c(2,:);
S=0.5.*(x2.*y3+x1.*y2+x3.*y1-x2.*y1-x1.*y3-x3.*y2);
S_sqrt=2.*sqrt(S);
Dx=[(y2-y3)./S_sqrt;(y3-y1)./S_sqrt;(y1-y2)./S_sqrt];
Dy=[(x3-x2)./S_sqrt;(x1-x3)./S_sqrt;(x2-x1)./S_sqrt];
DiDj_x=[Dx([1 2 3],:).*Dx([1 2 3],:);Dx([1 2 3],:).*Dx([2 3 1],:);Dx([1 2 3],:).*Dx([3 1 2],:)];
DiDj_y=[Dy([1 2 3],:).*Dy([1 2 3],:);Dy([1 2 3],:).*Dy([2 3 1],:);Dy([1 2 3],:).*Dy([3 1 2],:)];
DiDj=reshape((DiDj_x+DiDj_y),1,9*nel);
ii=reshape([nd;nd;nd],1,9*nel);
jj=reshape([nd([1 2 3],:);nd([2 3 1],:);nd([3 1 2],:)],1,9*nel);
A=sparse(ii,jj,DiDj,sDof,sDof); 
end
function [B]=stiffnessMatrix(f,p,t,Driv)
%% Generate stiffness matrix
sDof=size(p,2);
nel=size(t,2);
nd=t(1:3,:);
a=p(:,nd(1,:));
b=p(:,nd(2,:));
c=p(:,nd(3,:));
x1=a(1,:); y1=a(2,:);
x2=b(1,:); y2=b(2,:);
x3=c(1,:); y3=c(2,:);

S=0.5.*(x2.*y3+x1.*y2+x3.*y1-x2.*y1-x1.*y3-x3.*y2);
S_sqrt=2.*sqrt(S);
Points = [0.816847572980459 0.091576213509771 0.091576213509771;
              0.091576213509771 0.816847572980459 0.091576213509771;
              0.091576213509771 0.091576213509771 0.816847572980459;
              0.108103018168070 0.445948490915965 0.445948490915965;
              0.445948490915965 0.108103018168070 0.445948490915965;
              0.445948490915965 0.445948490915965 0.108103018168070];
          
Weights = [0.109951743655322,0.109951743655322,0.109951743655322,...
               0.223381589678011,0.223381589678011,0.223381589678011];
pxy = [1;1;1;1;1;1];
x = pxy*x1+Points(:,2)*(x2-x1)+Points(:,3)*(x3-x1);
y = pxy*y1+Points(:,2)*(y2-y1)+Points(:,3)*(y3-y1);
ii = reshape([nd;nd;nd],1,9*nel);
jj = reshape([nd([1 2 3],:);nd([2 3 1],:);nd([3 1 2],:)],1,9*nel);
Matrixf = repmat(f(x,y),[1 1 3]);

if Driv==0
    L(1:6,:,1) = pxy*(1./S_sqrt).*(pxy*(x2.*y3-x3.*y2)+(pxy*(y2-y3)).*x+(pxy*(x3-x2)).*y);
    L(1:6,:,2) = pxy*(1./S_sqrt).*(pxy*(x3.*y1-x1.*y3)+(pxy*(y3-y1)).*x+(pxy*(x1-x3)).*y);
    L(1:6,:,3) = pxy*(1./S_sqrt).*(pxy*(x1.*y2-x2.*y1)+(pxy*(y1-y2)).*x+(pxy*(x2-x1)).*y);
    
    LiLj(1:6,:,1:3,1) = L(1:6,:,[1 2 3]).*L(1:6,:,[1 2 3]).*Matrixf;
    LiLj(1:6,:,1:3,2) = L(1:6,:,[1 2 3]).*L(1:6,:,[2 3 1]).*Matrixf;
    LiLj(1:6,:,1:3,3) = L(1:6,:,[1 2 3]).*L(1:6,:,[3 1 2]).*Matrixf;
    LiLj = [Weights*LiLj(1:6,:,1,1);Weights*LiLj(1:6,:,2,1);Weights*LiLj(1:6,:,3,1);...
          Weights*LiLj(1:6,:,1,2);Weights*LiLj(1:6,:,2,2);Weights*LiLj(1:6,:,3,2);...
          Weights*LiLj(1:6,:,1,3);Weights*LiLj(1:6,:,2,3);Weights*LiLj(1:6,:,3,3)]; 
    LiLj=reshape(LiLj,1,9*nel);
    B=sparse(ii,jj,LiLj,sDof,sDof); 
elseif Driv==1
    Dx=[(y2-y3)./S_sqrt;(y3-y1)./S_sqrt;(y1-y2)./S_sqrt];
    Dy=[(x3-x2)./S_sqrt;(x1-x3)./S_sqrt;(x2-x1)./S_sqrt];
    DiDj_x=[Dx([1 2 3],:).*Dx([1 2 3],:);Dx([1 2 3],:).*Dx([2 3 1],:);Dx([1 2 3],:).*Dx([3 1 2],:)];
    DiDj_y=[Dy([1 2 3],:).*Dy([1 2 3],:);Dy([1 2 3],:).*Dy([2 3 1],:);Dy([1 2 3],:).*Dy([3 1 2],:)];
    
    int_f = repmat(Weights*f(x,y),9,1);
    DiDj=reshape((DiDj_x+DiDj_y).*int_f,1,9*nel);
    B=sparse(ii,jj,DiDj,sDof,sDof); 
end
end
function [F]=rightHand(f,p,t)
%% Generate right hands: Fi= [f,φi]
sDof=size(p,2);
nel=size(t,2);
nd=t(1:3,:);
a=p(:,nd(1,:));
b=p(:,nd(2,:));
c=p(:,nd(3,:));
x1=a(1,:); y1=a(2,:);
x2=b(1,:); y2=b(2,:);
x3=c(1,:); y3=c(2,:);

Points = [0.816847572980459 0.091576213509771 0.091576213509771;
              0.091576213509771 0.816847572980459 0.091576213509771;
              0.091576213509771 0.091576213509771 0.816847572980459;
              0.108103018168070 0.445948490915965 0.445948490915965;
              0.445948490915965 0.108103018168070 0.445948490915965;
              0.445948490915965 0.445948490915965 0.108103018168070];
          
Weights = [0.109951743655322,0.109951743655322,0.109951743655322,...
               0.223381589678011,0.223381589678011,0.223381589678011];

pxy = [1;1;1;1;1;1];
x = pxy*x1+Points(:,2)*(x2-x1)+Points(:,3)*(x3-x1);
y = pxy*y1+Points(:,2)*(y2-y1)+Points(:,3)*(y3-y1);

L(1:6,:,1) = 1./2.*(pxy*(x2.*y3-x3.*y2)+(pxy*(y2-y3)).*x+(pxy*(x3-x2)).*y);
L(1:6,:,2) = 1./2.*(pxy*(x3.*y1-x1.*y3)+(pxy*(y3-y1)).*x+(pxy*(x1-x3)).*y);
L(1:6,:,3) = 1./2.*(pxy*(x1.*y2-x2.*y1)+(pxy*(y1-y2)).*x+(pxy*(x2-x1)).*y);
fLi = [Weights*(L(1:6,:,1).*f(x,y));Weights*(L(1:6,:,2).*f(x,y));Weights*(L(1:6,:,3).*f(x,y))];
fLi = reshape(fLi,1,3*nel);
ii = reshape(nd,1,3*nel);
F = accumarray(ii',fLi',[sDof,1]);
end
function [new_Un, new_Yn, new_Pn] = subQP(TOF, g_1, g_2, P, T, B, Am, Un, Yn, Pn, ua, ub, E_omega, lambda, Yd, inter_iter)
%% SQP begin
    t0 = tic;
    disp(['内层第', num2str(inter_iter), '次循环开始 ...'])
    Nx = sqrt(size(Yn, 1)) - 1;
%--------------------------------------------------------------------------
% Matrix   Aij = [▽φj,▽φi] + [(1+3Yn^2)φj,φi]
%--------------------------------------------------------------------------     
    A_cof = 1 + 3*Yn.^2;
    [r, c] = ndgrid(g_1 : (g_2/Nx) : g_2);
    matrix_A_cof = reshape(A_cof, Nx + 1, Nx + 1)';
    A_q = griddedInterpolant(r, c, matrix_A_cof, 'spline');%插值结果
    A = Am + stiffnessMatrix(A_q,P,T,0); 
%--------------------------------------------------------------------------
% Matrix Bij = [φj,φi] and vector Zi = [2Yn^3+EΩ,φi]
%--------------------------------------------------------------------------     
    Z_cof = 2*Yn.^3 + E_omega;
    matrix_Z_cof = reshape(Z_cof, Nx + 1, Nx + 1)';
    Z_f = griddedInterpolant(r, c, matrix_Z_cof, 'spline');
    Z = rightHand(Z_f, P, T);
%--------------------------------------------------------------------------
% Matrix  D2ij = [(PnYn)φj,φi]
%--------------------------------------------------------------------------     
    D2_cof = Yn .* Pn;
    matrix_D2_cof = reshape(D2_cof, Nx + 1, Nx + 1)';
    D2_f = griddedInterpolant(r, c, matrix_D2_cof, 'spline');
    D2 = stiffnessMatrix(D2_f,P,T,0); 
%--------------------------------------------------------------------------
% Matrix B1ij = [(1-6PnYn)φj,φi] and vector Z1i = [6PnYn^2-Yd,φi]
%--------------------------------------------------------------------------     
    B1 = B - 6*D2;
    Z1 = 6*D2*Yn - B*Yd;
%--------------------------------------------------------------------------
% Box constraint 
%--------------------------------------------------------------------------   
    Ua = ua*ones(size(Yn, 1), 1);
    Ub = ub*ones(size(Yn, 1), 1);
%--------------------------------------------------------------------------
% PDAS 
%--------------------------------------------------------------------------   
    t1 = tic;
if TOF == 0
    [U_N, Y_N, P_N, p] = uypPDAS(Un, Ua, Ub, A, B, Z, B1, Z1, lambda, 1e-10);
    countTime5 = toc(t0);
    uypPDAS_time = toc(t1);
    disp(['！uypPDAS 计算完成, 总迭代步数', num2str(p), ', 用时', num2str(uypPDAS_time), '秒, 累计用时', num2str(countTime5), '秒.'])
elseif TOF == 1    
    [U_N, Y_N, P_N, pI] = uypPDAS_k(Un, Ua, Ub, A, B, Z, B1, Z1, lambda, inter_iter);
    countTime5 = toc(t0);
    uypPDAS_I_time = toc(t1);
    disp(['！uypPDAS_I 计算完成, 总迭代步数', num2str(pI), ', 用时', num2str(uypPDAS_I_time), '秒, 累计用时', num2str(countTime5), '秒.'])
end
    new_Un = U_N;
    new_Yn = Y_N;
    new_Pn = P_N;
    countTime6 = toc(t0);
    disp(['！生成完成, 累计用时', num2str(countTime6), '秒.'])
    disp(' ')
end
function [U_N, Y_N, P_N, p] = uypPDAS(U0, Ua, Ub, A, B, Z, B1, Z1, lambda, err)
%% PDAS
    n = size(Ua,1);
    p = 0;
    U_N = zeros(n,1);
    Y_N = zeros(n,1);
    P_N = zeros(n,1);

    MUa = lambda*(Ua - U0); 
    MUb = lambda*(U0 - Ub); 
    ca = lambda;
    cb = lambda;

    while 1  
    % Active sets: Ana, Anb;  Inactive set：In
        Ana = find(ca * U0 - MUa < ca * Ua);
        Anb = find(cb * U0 + MUb > cb * Ub);
        In = find((ca * U0 - MUa >= ca * Ua) & (cb * U0 + MUb <= cb * Ub));

        Areo = [A(In, In), A(In, Ana), A(In, Anb); ...
             A(Ana, In), A(Ana, Ana), A(Ana, Anb); ...
             A(Anb, In), A(Anb, Ana), A(Anb, Anb)];
        Badj = [B1(In, In), B1(In, Ana), B1(In, Anb); ...
              B1(Ana, In), B1(Ana, Ana), B1(Ana, Anb); ...
              B1(Anb, In), B1(Anb, Ana), B1(Anb, Anb)];
        Z1reo = [Z1(In); Z1(Ana); Z1(Anb)];
        Bsta = [(1/lambda) * B(In, In) 0*B(In, Ana) 0*B(In, Anb); ...
              (1/lambda) * B(Ana, In) 0*B(Ana, Ana) 0*B(Ana, Anb); ...
              (1/lambda) * B(Anb, In) 0*B(Anb, Ana) 0*B(Anb, Anb)]; 
        Zsta = [Z(In) + B(In, Ana)*Ua(Ana) + B(In, Anb)*Ub(Anb); ...
              Z(Ana) + B(Ana, Ana)*Ua(Ana) + B(Ana, Anb)*Ub(Anb); ...
              Z(Anb) + B(Anb, Ana)*Ua(Ana) + B(Anb, Anb)*Ub(Anb)];

        M = [Areo, Bsta; -Badj, Areo];
        R = [Zsta; Z1reo];
        V_N = M\R;
        
        Y_N(In) = V_N(1 : size(In,1));
        Y_N(Ana) = V_N(size(In,1)+1 : size(In,1)+size(Ana,1));
        Y_N(Anb) = V_N(size(In,1)+size(Ana,1)+1 : n);
        P_N(In) = V_N(n+1 : n+size(In,1));
        P_N(Ana) = V_N(n+1+size(In,1) : n+size(In,1)+size(Ana,1));
        P_N(Anb) = V_N(n+1+size(In,1)+size(Ana,1) : 2*n);
        U_N(Ana) = Ua(Ana);
        U_N(Anb) = Ub(Anb);
        U_N(In) = - (1/lambda)*P_N(In);

        MUa0 = MUa;
        MUb0 = MUb; 
        MUa(In) = 0;
        MUa(Anb) = 0;
        MUb(In) = 0;
        MUb(Ana) = 0;
        MUa(Ana) = P_N(Ana)+lambda*Ua(Ana);
        MUb(Anb) = - (P_N(Anb)+lambda*Ub(Anb));
        p=p+1;
        N0=[(U0'*B*U0 + U_N'*B*U_N - 2*U0'*B*U_N)^(1/2) (MUa0'*B*MUa0 + MUa'*B*MUa - 2*MUa0'*B*MUa)^(1/2) (MUb0'*B*MUb0 + MUb'*B*MUb - 2*MUb0'*B*MUb)^(1/2)];
        if  max(N0) <= err || p > 200 
            break;
        end
        U0 = U_N;
    end
end
function [U_N, Y_N, P_N, p] = uypPDAS_k(U0, Ua, Ub, A, B, Z, B1, Z1, lambda, k)
%% PDAS (inexact SQP)
    n = size(Ua,1);
    p = 0;
    U_N = zeros(n,1);
    Y_N = zeros(n,1);
    P_N = zeros(n,1);

    MUa = lambda*(Ua - U0); 
    MUb = lambda*(U0 - Ub); 
    ca = lambda;
    cb = lambda;

    while 1  
    % Active sets: Ana, Anb;  Inactive set：In
        Ana = find(ca * U0 - MUa < ca * Ua);
        Anb = find(cb * U0 + MUb > cb * Ub);
        In = find((ca * U0 - MUa >= ca * Ua) & (cb * U0 + MUb <= cb * Ub));

        Areo = [A(In, In), A(In, Ana), A(In, Anb); ...
             A(Ana, In), A(Ana, Ana), A(Ana, Anb); ...
             A(Anb, In), A(Anb, Ana), A(Anb, Anb)];
        Badj = [B1(In, In), B1(In, Ana), B1(In, Anb); ...
              B1(Ana, In), B1(Ana, Ana), B1(Ana, Anb); ...
              B1(Anb, In), B1(Anb, Ana), B1(Anb, Anb)];
        Z1reo = [Z1(In); Z1(Ana); Z1(Anb)];
        Bsta = [(1/lambda) * B(In, In) 0*B(In, Ana) 0*B(In, Anb); ...
              (1/lambda) * B(Ana, In) 0*B(Ana, Ana) 0*B(Ana, Anb); ...
              (1/lambda) * B(Anb, In) 0*B(Anb, Ana) 0*B(Anb, Anb)]; 
        Zsta = [Z(In) + B(In, Ana)*Ua(Ana) + B(In, Anb)*Ub(Anb); ...
              Z(Ana) + B(Ana, Ana)*Ua(Ana) + B(Ana, Anb)*Ub(Anb); ...
              Z(Anb) + B(Anb, Ana)*Ua(Ana) + B(Anb, Anb)*Ub(Anb)];
        M = [Areo, Bsta; -Badj, Areo];
        R = [Zsta; Z1reo];
        V_N = M\R;
        
        Y_N(In) = V_N(1 : size(In,1));
        Y_N(Ana) = V_N(size(In,1)+1 : size(In,1)+size(Ana,1));
        Y_N(Anb) = V_N(size(In,1)+size(Ana,1)+1 : n);
        P_N(In) = V_N(n+1 : n+size(In,1));
        P_N(Ana) = V_N(n+1+size(In,1) : n+size(In,1)+size(Ana,1));
        P_N(Anb) = V_N(n+1+size(In,1)+size(Ana,1) : 2*n);
        U_N(Ana) = Ua(Ana);
        U_N(Anb) = Ub(Anb);
        U_N(In) = - (1/lambda)*P_N(In);

        MUa0 = MUa;
        MUb0 = MUb; 
        MUa(In) = 0;
        MUa(Anb) = 0;
        MUb(In) = 0;
        MUb(Ana) = 0;
        MUa(Ana) = P_N(Ana)+lambda*Ua(Ana);
        MUb(Anb) = - (P_N(Anb)+lambda*Ub(Anb));

        p=p+1;
        N0=[(U0'*B*U0 + U_N'*B*U_N - 2*U0'*B*U_N)^(1/2) (MUa0'*B*MUa0 + MUa'*B*MUa - 2*MUa0'*B*MUa)^(1/2) (MUb0'*B*MUb0 + MUb'*B*MUb - 2*MUb0'*B*MUb)^(1/2)];
        if  max(N0) <= max( 1/(k)^(2.0)  ,1e-10)  || p > 200 
            break;
        end
        U0 = U_N;
    end
end



