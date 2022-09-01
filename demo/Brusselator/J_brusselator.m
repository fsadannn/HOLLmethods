function dfdt = J_brusselator( t, y, theta )
%J_BRUSELATOR Summary of this function goes here
%   Detailed explanation goes here
N = length(y)/2;
alphac = 0.02;

persistent ee;
persistent ee2;

[t1,t2] = size(y);
if t1<t2
    y = y(:);
end

if length(ee)~= N
    ee = (alphac*((N+1)*(N+1))).*ones(N,1);
    ee2 = -2.0.*ee;
end


uv2=2.0.*(y(1:N).*y(N+1:2*N));
u2=y(1:N).^2;

D1=spdiags([ee ee2+uv2-4.0 ee], -1:1, N, N);
D2=spdiags(u2, 0, N, N);
D3=spdiags(3.0-uv2, 0, N, N);
D4=spdiags([ee ee2-u2 ee], -1:1, N, N);

dfdt = [D1,D2;D3,D4];

end

