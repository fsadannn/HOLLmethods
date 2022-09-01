function [ jac ] = J_gs2d( t, y, theta )
%J_GS2D Summary of this function goes here
%   Detailed explanation goes here
Nd2 = length(y)/2;
N = sqrt(Nd2);
gsdu=0.2;
gsdv=0.1;
gsa=0.04;
gsb=0.06;
persistent GS_I1;
persistent GS_I2;
persistent GS_JAC;
persistent N2;

if length(GS_I1)~= Nd2
    N2 = N*N;
    GS_I1 = 1:N2;
    GS_I2 = N2+1:2*N2;
    e=-ones(N,1);
    T = spdiags([e,-2.0.*e,e],-1:1,N,N);
    T(1,2) = -2.0;  T(N,N-1) = -2.0;
    c = (N-1)^2;
    GS_JAC = c*(kron(T,speye(N,N)) + kron(speye(N,N),T));
end

tmp = 2.0*y(GS_I1).*y(GS_I2);
tmp2 = y(GS_I1).^2;
J21 = spdiags(y(GS_I1).^2,0,N2,N2);
J11 = -gsdu.*GS_JAC - spdiags(tmp2+gsa,0,N2,N2);
J12 = spdiags(-tmp,0,N2,N2);
J22 = -gsdv.*GS_JAC -spdiags(tmp-(gsa+gsb),0,N2,N2);

jac = [J11 J12; J21 J22];

end