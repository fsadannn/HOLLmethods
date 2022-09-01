function [phi, err, m, nexpo, breakdown, nfeval] = phi1LLRK4_hJ_f_freeJ_na(func,b,h,t,y,m, ...
    rtol,atol,kdmax,kdmin,gamma)
%PHI1LLDP_hJ_f Summary of this function goes here
%   Detailed explanation goes here
persistent V
persistent H
persistent mold

jorder=1;

n=length(b);
m=min(m,n);
if isempty(mold)
    mold = m;
end


fixpade = 3;

rfacmin=1;
fac=1/log(2);

nexpo=0;
ww = atol+rtol.*abs(y);


breakdown=0;
btol = 2*eps;
mb = m;
rndoff= eps;
k1 = 3;
hsubspace = h^(1.5);
delta = hsubspace;
delta1 = 1/delta;


% begin Arnoldi
if size(V,1)~=n
    V = zeros(n,m+1);
    H = zeros(m+3,m+3);
else
    if mold>m
        H = [H(1:m,1:m),zeros(m,3);zeros(3,m+3)];
    elseif  m>mold
        H = [H(1:mold,1:mold),zeros(mold,m-mold+3);zeros(m-mold+3,m+3)];
    end
    if size(V,2)<m+1
        V = [V,zeros(n,m-size(V,2)+1)];
    end
end


deltahsubspace = delta*hsubspace;

hd2 = h/2;

normb = norm(b);


% build base of hA and b

% begin Arnoldi
V(:,1)=(1/normb).*b;
% test if norm(b) is 0
if any(isnan(V(:,1)))
    phi = sparse(n,5);
    mold = 0;
    err = 0;
    warning('llint:phi1LLDP_hJ_f_na',...
       'First vector of krylov subspace seems to be 0.')
    return;
end
for j = 1:m
    p = delta1.*(func(t,y+deltahsubspace.*(V(:,j)))-b);
    s = V(:,1:j);
    H(1:j,j) = s.'*p;
    p = p - s*H(1:j,j);
    s = norm(p);
    if s < btol
        k1 = 0;
        mb = j;
        breakdown=1;
        break
    end
    H(j+1,j) = s;
    V(:,j+1) = (1/s).*p;
end
nfeval=jorder*mb;
% end Arnoldi
% using scaling invariance property of Arnoldi
% rescaling H to have H for the subspace of A and b
H=(1/hsubspace).*H;
% build \hat{H}
if k1 == 0
    if mb>1
        mb=mb-1;
        warning('llint:phi1LLDP_hJ_f_na',...
       'Breakdown at dimension 1.')
    end
    m=mb;
    hk = H(m+1,m);
    H=[H(1:m,1:m),zeros(m,3);zeros(3,m+3)];
else
    hk = H(m+1,m);
    H(m+1,m) = 0;
end
H(1,m+1) = 1;
H(m+1,m+2) = 1; H(m+2,m+3) = 1;
avm1dot = delta1.*(func(t,y+deltahsubspace.*(V(:,m+1)))-b);
nfeval=nfeval+jorder;

work=1;

while work
    % select p-p of Pade
    nnorm = norm(H,'inf');
    pd = fixpade;
    
    nhC = hd2*nnorm;
    [~,e] = log2(nhC);
    s = max(0,e+1);
    M12 = expm64v4(hd2.*H,pd,s);

    M1=M12*M12;
    
    nexpo = nexpo + 1;
    
    % calating \hat{E}
    beta = normb;
    %error relative
    % the divsion by h is because Av_{m+1} is in reality
    % hAv{m+1}(avm1dot) and need rescaling
    err_rel=sqrt((1/n)*sum((((hk*M1(m,m+3)*beta/h).*avm1dot)./(ww)).^2));
    if err_rel/gamma>=1 && m<kdmax
        rfacmax=max(1,m/3);
        knew =  log(err_rel/gamma)*fac;
        knew = ceil(m + min(rfacmax,max(knew,rfacmin)));
        knew = max(kdmin, min(kdmax,knew));

        H = [hsubspace.*H(1:m,1:m),zeros(m,knew-m+3);zeros(knew-m+3,knew+3)];
        H(m+1,m) = hk*hsubspace;
        if size(V,2)<knew+1
            V = [V,zeros(n,knew-size(V,2)+1)];
        end
        mtemp=m+1;
        m=knew;
        mb=m;
        k1 = 3;
        
        j=mtemp;
        s = V(:,1:j);
        H(1:j,j) = s.'*avm1dot;
        avm1dot = avm1dot - s*H(1:j,j);
        s = norm(avm1dot);
        if s < btol
            k1 = 0;
            mb = j;
            breakdown=1;
            mtemp=m;
        end
        H(j+1,j) = s;
        V(:,j+1) = (1/s)*avm1dot;

        for j = mtemp+1:m
            p = delta1.*(func(t,y+deltahsubspace.*(V(:,j)))-b);
            s = V(:,1:j);
            H(1:j,j) = s.'*p;
            p = p - s*H(1:j,j);
            s = norm(p);
            if s < btol
                k1 = 0;
                mb = j;
                breakdown=1;
                break;
            end
            H(j+1,j) = s;
            V(:,j+1) = (1/s)*p;
        end
        nfeval=nfeval+jorder*abs(mtemp+1-mb);
        H=(1/hsubspace).*H;
        % build \hat{H}
        if k1 == 0
            if mb>1
                mb=mb-1;
            end
            m=mb;
            hk = H(m+1,m);
            H=[H(1:m,1:m),zeros(m,3);zeros(3,m+3)];
        else
            hk = H(m+1,m);
            H(m+1,m) = 0;
        end
        H(1,m+1) = 1;
        H(m+1,m+2) = 1; H(m+2,m+3) = 1;
        avm1dot = delta1.*(func(t,y+deltahsubspace.*(V(:,m+1)))-b);
        nfeval=nfeval+jorder;
    else
        break;
    end
end

M1(m+1,m+1) = hk*M1(m,m+2);
M1(m+2,m+1) = hk*M1(m,m+3);
M12(m+1,m+1) = hk*M12(m,m+2);
M12(m+2,m+1) = hk*M12(m,m+3);

phi = zeros(n,2);
mx = m + 1;

avm1dot = V(:,1:mx);
% Matix projection
phi(:,1) = avm1dot*(beta*M12(1:mx,m+1));
phi(:,2) = avm1dot*(beta*M1(1:mx,m+1));

err = max(err_rel,rndoff);
mold = m;

end

