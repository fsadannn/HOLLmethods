function [ jac ] = J_bruss2d( t, y, theta )
%J_BRUSS2D Summary of this function goes here
%   Detailed explanation goes here
Nd2 = length(y)/2;
N = sqrt(Nd2);
alphac = 0.02;
persistent BRUSS_I1;
persistent BRUSS_I2;
persistent BRUSS_JAC;
persistent N2;

if length(BRUSS_I1)~= Nd2
    N2 = N*N;
    BRUSS_I1 = 1:N2;
    BRUSS_I2 = N2+1:2*N2;
    e=-ones(N,1);
    T = spdiags([e,-2.0.*e,e],-1:1,N,N);
    T(1,2) = -2.0;  T(N,N-1) = -2.0;
    c = alphac * (N-1)*(N-1);
    BRUSS_JAC = c*(kron(T,speye(N,N)) + kron(speye(N,N),T));
end

tmp = 2.0*y(BRUSS_I1).*y(BRUSS_I2);
J11 = -BRUSS_JAC + spdiags(tmp - 4.0,0,N2,N2);
J12 = spdiags(y(BRUSS_I1).^2,0,N2,N2);
J22 = -BRUSS_JAC -J12;
J21 = spdiags(3.0 - tmp,0,N2,N2);
jac = [J11 J12; J21 J22];

end