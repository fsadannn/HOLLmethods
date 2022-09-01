function [ dy ] = f_bruss2d( t, y, theta )
%F_BRUSS2D Summary of this function goes here
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
    T = spdiags([e,(-2.0).*e,e],-1:1,N,N);
    T(1,2) = -2.0;  T(N,N-1) = -2.0;
    c = alphac * (N-1)*(N-1);
    BRUSS_JAC = c.*(kron(T,speye(N,N)) + kron(speye(N,N),T));
end

dy = zeros(2*N2,1);
dy(BRUSS_I1) = -(BRUSS_JAC*y(BRUSS_I1));
dy(BRUSS_I2) = -(BRUSS_JAC*y(BRUSS_I2));
tmp = (y(BRUSS_I1).^2).*y(BRUSS_I2);
dy(BRUSS_I1) = 1 + dy(BRUSS_I1) + tmp - 4.0*y(BRUSS_I1);
dy(BRUSS_I2) = dy(BRUSS_I2) + 3.0*y(BRUSS_I1) - tmp;

end