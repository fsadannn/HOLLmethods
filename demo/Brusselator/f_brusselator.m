function dydt = f_brusselator( t, y, theta )
%F_BRUSEATOR Summary of this function goes here
%   Detailed explanation goes here
N = length(y)/2;
alphac = 0.02;

% if nargin == 2
%     theta = 0;
% end;

c = alphac *((N+1)*(N+1));

dydt = zeros(2*N,1);

i = 1;
i2 = N+1;
dydt(i) = 1 + y(i)^2*y(i2) - 4*y(i) + c*(1-2*y(i)+y(i+1));
dydt(i2) = 3*y(i) - y(i)^2*y(i2) + c*(3-2*y(i2)+y(i2+1));

% Evaluate the 2 components of the function at all interior grid points.
i = 2:N-1;
i2 = N+2:2*N-1;
u2v=(y(i).^2).*y(i2);
dydt(i) = 1 + u2v -4.*y(i) + c.*(y(i-1)-2.*y(i)+y(i+1));
dydt(i2) = 3.*y(i) - u2v + c.*(y(i2-1)-2.*y(i2)+y(i2+1));

% Evaluate the 2 components of the function at the other edge of the grid
% (with edge conditions).
i = N;
i2 = 2*N;
dydt(i) = 1 + y(i)^2*y(i2)-4*y(i) + c*(y(i-1)-2*y(i)+1);
dydt(i2) = 3*y(i) - y(i)^2*y(i2) + c*(y(i2-1)-2*y(i2)+3);
end

