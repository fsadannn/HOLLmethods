function y = JFLLRK(odeFcn,tt,yo,rtol,atol,kdim, pade, ...
    order_out, order_in, proportion_out, proportion_in, proportion_in_sub)
%
%  not adaptive implementation of the explicit Jacobian-free Locally Linearized Runge-Kutta scheme
%  with variable parameters
%
% inputs
%          odeFcn: right hand side differential equation function
%              tt: integration times
%              yo: intital value
%            rtol: relative tolerance
%            atol: absolute tolerance
%            kdim: Krylov subspace dimension 
%            pade: pade order
%       order_out: finite difference order for approximating the jacobian times vector 
%                  in the non linear equation
%        order_in: finite difference order for approximating the jacobian times vector 
%                  in the linear equation
%  proportion_out: power of h to take as delta in the finite difference
%                  of the non linear equation
%   proportion_in: power of h to take as delta in the finite difference
%                  of the lineal equation
%   proportion_in: power of h to compute the Krylove subspace (\tau)
%
% output
%               y: solution
%

d=length(yo);
n=length(tt);
yi=yo(:);
yy=zeros(d,n);
yy(:,1)=yi;
hh=diff(tt);

% THE MAIN LOOP

gamma = 0.005;

FcnUsed = isa(odeFcn,'function_handle');
odeFcn_main = odefcncleanup(FcnUsed,odeFcn,{});

Fxphi = zeros(d,2);


for i=1:n-1
    t=tt(i);
    h=hh(i);

    f1 = odeFcn_main(t, yi);
  
    

    [phi,~,~,~,~,~] = phi1LLRK4_hJ_f_freeJ_na_all_pade_test(odeFcn_main,f1,h,t,yi,kdim,...
        rtol,atol,kdim,kdim,gamma,order_in,pade, proportion_in, proportion_in_sub);


    if order_out==1
        nyp = h^(proportion_out);
        Fxphi(:,1) = (odeFcn_main(t,yi+nyp*phi(:,1))-f1)./nyp;
        Fxphi(:,2) = (odeFcn_main(t,yi+nyp*phi(:,2))-f1)./nyp;
    elseif order_out==2
        nyp = h^(proportion_out);
        Fxphi(:,1) = (odeFcn_main(t,yi+nyp*phi(:,1))-odeFcn_main(t,yi-nyp*phi(:,1)))./(2*nyp);
        Fxphi(:,2) = (odeFcn_main(t,yi+nyp*phi(:,2))-odeFcn_main(t,yi-nyp*phi(:,2)))./(2*nyp);
    elseif order_out==3
        nyp = h^(proportion_out);
        m3f1 = -3.*f1;
        Fxphi(:,1) = (m3f1 + 4.*odeFcn_main(t,yi+nyp*phi(:,1)) -odeFcn_main(t,yi+(2*nyp).*phi(:,1)))./(2*nyp);
        Fxphi(:,2) = (m3f1 + 4.*odeFcn_main(t,yi+nyp*phi(:,2)) -odeFcn_main(t,yi+(2*nyp).*phi(:,2)))./(2*nyp);
    end

    yLL2 = yi + phi(:,2);

    y2 = yi + phi(:,1);
    t2 = t + h/2;
    f2 = odeFcn_main(t2, y2)- f1 - Fxphi(:,1);

    y3 = yi + (h/2).*f2 + phi(:,1);
    t3 = t + h/2;
    f3 = odeFcn_main(t3, y3)- f1 - Fxphi(:,1);

    y4 = yi + h.*f3  +phi(:,2);
    t4 = t + h;
    f4 = odeFcn_main(t4, y4)- f1 - Fxphi(:,2);

    yi = yLL2 + (h/6).*(2.*f2+2.*f3+f4);
    yy(:,i+1)=yi;

end
y=yy';
end
