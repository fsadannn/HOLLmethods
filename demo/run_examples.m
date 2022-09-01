disp('Warning: ');
disp('  If the user does not have license for the Parallel Computing Toolbox,') 
disp('  replace the "parfor" by "for" in the files ')
disp('  convergence_order_test_fj_m_p.m, convergence_order_test_fj_in.m, convergence_order_test_fj_m_p.m')
disp('  ')


abspath = which('run_examples');
pos = strfind(abspath, filesep); pos = pos(end);
abspath = abspath(1:pos - 1);

cd(sprintf('%s%s%s',abspath,filesep,'..'));

s = [
 [abspath,filesep,'..',filesep,'demo;']...,
 [abspath,filesep,'..',filesep,'llint;']...,
 [abspath,filesep,'Brusselator2D;']...,
 [abspath,filesep,'Burgers;']...,
 [abspath,filesep,'GrayScott2D;']...,
 [abspath,filesep,'Brusselator;']
];

path(s,path);
initllPaths(true);


fname = @f_brusselator;
N = 100;
x0=[1+sin((2*pi/(N+1))*(1:N)),repmat(3,1,N)];
IT=[0 1];
h=[2.5e-3,5e-3,0.01,0.02,0.025];
h=h(end:-1:1);

t_steps = zeros(1,length(h));
t_nfevals = zeros(1,length(h));
t_NofKry = zeros(1,length(h));
t_ksum = zeros(1,length(h));
t_kmin = zeros(1,length(h));
t_kmax = zeros(1,length(h));
t_nexpo = zeros(1,length(h));

parfor i=1:length(h)
    time = IT(1):h(i):IT(2);
    NofKry = max(size(time))-1;
    steps=  max(size(time))-1;
    
    [y,kmin,kmax,ksum,nexpo,~, nfevals]= JFLLRK4(fname,time,x0);
    
    t_steps(i)=steps;
    t_nfevals(i)=nfevals;
    t_NofKry(i)=NofKry
    t_ksum(i)=ksum
    t_kmin(i)=kmin;
    t_kmax(i)=kmax;
    t_nexpo(i)=nexpo;
end
disp(' ');
disp('Performance of the JFLLRK4 in Table 5 (Brusselator)');
Results.h=h';
Results.Steps=t_steps';
Results.f_Eval=t_nfevals';
Results.K_spaces=t_NofKry';
Results.Pade=t_nexpo';
Results.m_total=t_ksum';
Results.m_min=t_kmin';
Results.m_max=t_kmax';
TableT = struct2table(Results);
disp(TableT);



fname = @f_bruss2d;
N = 40;
N2 = N*N;
BRUSS_I1 = 1:N2;
BRUSS_I2 = N2+1:2*N2;
[X,Y] = meshgrid(linspace(0,1,N),linspace(0,1,N));
x0(BRUSS_I1) = 1+sin((2*pi).*X(:)).*sin((2*pi).*Y(:));
x0(BRUSS_I2) = 3;
clear X;
clear Y;
clear BRUSS_I1;
clear BRUSS_I2;
IT=[0 0.1];
h=[0.002,0.0025,0.005,0.00625,0.01];
h=h(end:-1:1);

t_steps = zeros(1,length(h));
t_nfevals = zeros(1,length(h));
t_NofKry = zeros(1,length(h));
t_ksum = zeros(1,length(h));
t_kmin = zeros(1,length(h));
t_kmax = zeros(1,length(h));
t_nexpo = zeros(1,length(h));

parfor i=1:length(h)
    time = IT(1):h(i):IT(2);
    NofKry = max(size(time))-1;
    steps=  max(size(time))-1;
    
    [y,kmin,kmax,ksum,nexpo,~, nfevals]= JFLLRK4(fname,time,x0);
    
    t_steps(i)=steps;
    t_nfevals(i)=nfevals;
    t_NofKry(i)=NofKry
    t_ksum(i)=ksum
    t_kmin(i)=kmin;
    t_kmax(i)=kmax;
    t_nexpo(i)=nexpo;
end
disp('Performance of the JFLLRK4 in Table 6 (Brusselator 2D)');
Results.h=h';
Results.Steps=t_steps';
Results.f_Eval=t_nfevals';
Results.K_spaces=t_NofKry';
Results.Pade=t_nexpo';
Results.m_total=t_ksum';
Results.m_min=t_kmin';
Results.m_max=t_kmax';
TableT = struct2table(Results);
disp(TableT);



fname = @f_burgers;
N = 400;
x0=[((sin((3*pi/(N+1)).*(1:N))).^2).*((1-1/(N+1).*(1:N)).^(3/2))];
IT=[0 0.5];
h=[3.1250e-04,6.25e-4,1.25e-3,2.5e-3,5e-3];
h=h(end:-1:1);

t_steps = zeros(1,length(h));
t_nfevals = zeros(1,length(h));
t_NofKry = zeros(1,length(h));
t_ksum = zeros(1,length(h));
t_kmin = zeros(1,length(h));
t_kmax = zeros(1,length(h));
t_nexpo = zeros(1,length(h));

parfor i=1:length(h)
    time = IT(1):h(i):IT(2);
    NofKry = max(size(time))-1;
    steps=  max(size(time))-1;
    
    [y,kmin,kmax,ksum,nexpo,~, nfevals]= JFLLRK4(fname,time,x0);
    
    t_steps(i)=steps;
    t_nfevals(i)=nfevals;
    t_NofKry(i)=NofKry
    t_ksum(i)=ksum
    t_kmin(i)=kmin;
    t_kmax(i)=kmax;
    t_nexpo(i)=nexpo;
end
disp('Performance of the JFLLRK4 in Table 7 (Burger)');
Results.h=h';
Results.Steps=t_steps';
Results.f_Eval=t_nfevals';
Results.K_spaces=t_NofKry';
Results.Pade=t_nexpo';
Results.m_total=t_ksum';
Results.m_min=t_kmin';
Results.m_max=t_kmax';
TableT = struct2table(Results);
disp(TableT);



fname = @f_gs2d;
N = 20;
N2 = N*N;
GS_I1 = 1:N2;
GS_I2 = N2+1:2*N2;
[X,Y] = meshgrid(linspace(0,1,N),linspace(0,1,N));
x0(GS_I1) = 1-exp(-150.*(X(:)-1/2).^2+(Y(:)-1/2).^2);
x0(GS_I2) = exp(-150.*(X(:)-1/2).^2+2.*(Y(:)-1/2).^2);
clear X;
clear Y;
clear GS_I1;
clear GS_I2;
IT=[0 0.1];
h=[0.00125,0.002,0.0025,0.005,0.01];
h=h(end:-1:1);

t_steps = zeros(1,length(h));
t_nfevals = zeros(1,length(h));
t_NofKry = zeros(1,length(h));
t_ksum = zeros(1,length(h));
t_kmin = zeros(1,length(h));
t_kmax = zeros(1,length(h));
t_nexpo = zeros(1,length(h));

parfor i=1:length(h)
    time = IT(1):h(i):IT(2);
    NofKry = max(size(time))-1;
    steps=  max(size(time))-1;
    
    [y,kmin,kmax,ksum,nexpo,~, nfevals]= JFLLRK4(fname,time,x0);
    
    t_steps(i)=steps;
    t_nfevals(i)=nfevals;
    t_NofKry(i)=NofKry
    t_ksum(i)=ksum
    t_kmin(i)=kmin;
    t_kmax(i)=kmax;
    t_nexpo(i)=nexpo;
end
disp('Performance of the JFLLRK4 in Table 8 (Gray-Scott 2D)');
Results.h=h';
Results.Steps=t_steps';
Results.f_Eval=t_nfevals';
Results.K_spaces=t_NofKry';
Results.Pade=t_nexpo';
Results.m_total=t_ksum';
Results.m_min=t_kmin';
Results.m_max=t_kmax';
TableT = struct2table(Results);
disp(TableT);

