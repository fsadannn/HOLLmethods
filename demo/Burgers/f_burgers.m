function dydt = f_burgers( t, y, theta )
%F_BURGES Summary of this function goes here
%   Detailed explanation goes here
N = length(y);
nuc = 3e-4;

n1 = N+1;
n14 = n1/4;
nucn12 = nuc*n1*n1;
dydt = zeros(N,1);

dydt(1) = -n14*y(2)*y(2) + nucn12*(-2*y(1)+y(2));

index=3:N;
index2=2:N-1;
index3=1:N-2;
dydt(index2) = -n14.*(y(index).*y(index)-y(index3).*y(index3)) + ...
        nucn12.*(y(index3)-2.*y(index2)+y(index));

dydt(N) = n14*y(N-1)*y(N-1) + nucn12*(y(N-1)-2*y(N));

end

