function [y,kmin,kmax,ksum,nexpo,breakcont, nfevals]  = JFLLRK4(odeFcn,tt,yo,rtol,atol)
%
% not adaptive implementation of the explicit Jacobian-free Locally Linearized Runge-Kutta scheme of order 4
%
% inputs
%     odeFcn: right hand side differential equation function
%         tt: integration times
%         yo: intital value
%       rtol: relative tolerance
%       atol: absolute tolerance
%
% outputs
%          y: solution
%       kmin: minimum Krylov subspace dimension
%       kmax: maximum Krylov subspace dimension
%       ksum: total number of Krylov subspace dimension
%      nexpo: number of exponential matrix
%  breakcont: breakdown counter
%    nfevals: number of function evaluations

kdmax = 30;
kdmin = 4;
d=length(yo);
n=length(tt);
yi=yo(:);
yy=zeros(d,n);
yy(:,1)=yi;
hh=diff(tt);

if nargin < 5
    rtol=1e-9;
    atol=1e-12;
end

% THE MAIN LOOP
%krilov consants and parameters
gamma = 0.005;
kdmax = min(kdmax,d);
%debug
ksum = 0;
kmin = kdmax;
kmax = kdmin;
nexpo=0;
breakcont=0;
%adaptive for krilov
kdim=kdmin;
fac=1/log(2);
nfevals = 0;
FcnUsed = isa(odeFcn,'function_handle');
odeFcn_main = odefcncleanup(FcnUsed,odeFcn,{});

for i=1:n-1
    t=tt(i);
    h=hh(i);

    f1 = odeFcn_main(t, yi);

    [phi,kerror,kdim,nexpcont,~,knfevals] = phi1LLRK4_hJ_f_freeJ_na(odeFcn_main,f1,h,t,yi,kdim,...
            rtol,atol,kdim,kdim,gamma);
    %information
    ksum = ksum + kdim;
    nfevals = nfevals + knfevals;
    nexpo = nexpo + nexpcont;
    kmin=min(kmin,kdim);
    kmax=max(kmax,kdim);

    nyp1=sqrt(h);
    nyp2=nyp1;
    
    phi1 = phi(:,1);
    phi2 = phi(:,2);

     m3f1 = -3.*f1;
    deltav=nyp1.*phi1;
    Fxphi1 = (1/(2*nyp1)).*(m3f1 + 4.*odeFcn_main(t,yi+deltav) -odeFcn_main(t,yi+2.*deltav));
    deltav=nyp2.*phi2;
    Fxphi2 = (1/(2*nyp2)).*(m3f1 + 4.*odeFcn_main(t,yi+deltav) -odeFcn_main(t,yi+2.*deltav));
    
    nfevals = nfevals + 4;

    yLL2 = yi + phi2;

    y2 = yi + phi1;
    t2 = t + h/2;
    f2 = odeFcn_main(t2, y2)- f1 - Fxphi1;

    y3 = yi + (h/2).*f2 + phi1;
    t3 = t + h/2;
    f3 = odeFcn_main(t3, y3)- f1 - Fxphi1;

    y4 = yi + h.*f3  + phi2;
    t4 = t + h;
    f4 = odeFcn_main(t4, y4)- f1 - Fxphi2;

    yi = yLL2 + (h/6).*(2.*f2+2.*f3+f4);
    yy(:,i+1)=yi;
    nfevals = nfevals + 4;

    afacmax=-kdim/4;
    afacmin=kdim/3;
    kdnew =  log(kerror/gamma)*fac;
    kdnew = floor(kdim + max(afacmax,min(kdnew,afacmin)));
    kdim=max(kdmin,min(kdmax,kdnew));

end
y=yy';
end
