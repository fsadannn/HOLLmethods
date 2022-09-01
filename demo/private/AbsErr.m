function E = AbsErr(Y,Z)

% Y : vector con valores exactos
% Z : vector con valores approximados

dim=size(Y,1).*size(Y,2);
y=reshape(Y,dim,1);
z=reshape(Z,dim,1);
index=find(abs(y)>eps);
E = max(max(abs((y(index)-z(index)))));

