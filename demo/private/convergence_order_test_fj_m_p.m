clear all
% close all
N = 50;

fname = @f_brusselator;
Jname = @J_brusselator;
x0= [1+sin((2*pi/(N+1))*(1:N)),repmat(3,1,N)];
IT = [0,1];

options15s = odeset('RelTol',1.0e-12,'AbsTol',1.0e-14,'Jacobian',Jname);

kdim = 20;
pade = 2;
order_out = 2;
order_in = 2;
proportion_out = 0.5;
proportion_in = 0.5;
proportion_in_sub = 1;
algo = 2;

figure
ax1 = subplot(1,2,1,'XScale', 'log', 'YScale', 'log');
ax2 = subplot(1,2,2,'XScale', 'log', 'YScale', 'log');
hold(ax1,'on')
hold(ax2,'on')
box(ax1,'on')
box(ax2,'on')

m=1:3;

interval1 = 2.^(-(8:11));
interval2 = 2.^(-(7:10));
e4 = zeros(1,length(interval1));

Result2L.p=[2; 2; 2];
Result2L.m=[1; 2; 3];
Result2L.r=[2; 3; 4];
Result2L.est_r=[];
Result2L.ci=[];
for j=1:length(m)
    kdim = m(j);
    if m(j)<3
        interval = interval1;  
    else
        interval = interval2;  
    end
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out,proportion_in,...
             proportion_in_sub);

        [~,sol] = ode15s(fname,t,x0,options15s);

        e4(i)=AbsErr(sol,y4);
    end


    linterval4 = log10(interval);
    le4 = log10(e4);
    [c4,S4] = polyfit(linterval4,le4,1);
    pe4 = polyval(c4,linterval4,S4);

    iR4 = S4.R\eye(2);
    cov4 = (iR4*iR4').*(S4.normr^2/S4.df);

    plot(ax1,10.^linterval4, 10.^le4, '--o',10.^linterval4, 10.^pe4, '-' );
    % 90 percentil
    percentilev4=tinv(0.95,S4.df-1);

    confidence4 = percentilev4*sqrt(cov4(1,1)/(S4.df));

    Result2L.est_r=[Result2L.est_r; c4(1)];
    Result2L.ci=[Result2L.ci; confidence4];

end
disp('Table 2, left')
Table2L = struct2table(Result2L);
disp(Table2L);


kdim =3;
pade = 2;
order_out = 2;
order_in = 2;
proportion_out = 0.5;
proportion_in = 0.5;
proportion_in_sub = 1;
algo = 2;

pp=1:3;

interval1 = 2.^(-(8:11));
interval2 = 2.^(-(7:10));
e4 = zeros(1,length(interval1));

Result2R.m=[3; 3; 3];
Result2R.p_plus_p=[2; 4; 6];
Result2R.r=[2; 4; 4];
Result2R.est_r=[];
Result2R.ci=[];
for j=1:length(pp)
    pade = pp(j);
    if pp(j)<2
        interval = interval1;  
    else
        interval = interval2;  
    end
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out,proportion_in,...
             proportion_in_sub);

        [~,sol] = ode15s(fname,t,x0,options15s);

        e4(i)=AbsErr(sol,y4);
    end


    linterval4 = log10(interval);
    le4 = log10(e4);
    [c4,S4] = polyfit(linterval4,le4,1);
    pe4 = polyval(c4,linterval4,S4);

    iR4 = S4.R\eye(2);
    cov4 = (iR4*iR4').*(S4.normr^2/S4.df);

    plot(ax2,10.^linterval4, 10.^le4, '--o',10.^linterval4, 10.^pe4, '-' );
    % 90 percentil
    percentilev4=tinv(0.95,S4.df-1);
    
    confidence4 = percentilev4*sqrt(cov4(1,1)/(S4.df));
    
    Result2R.est_r=[Result2R.est_r; c4(1)];
    Result2R.ci=[Result2R.ci; confidence4];
end
disp(' ')
disp('Table 2, right')
Table2R = struct2table(Result2R);
disp(Table2R);


xlm1 = xlim(ax1);
xlm1 = xlm1(1);
xlm2 = xlim(ax2);
xlm2 = xlm2(1);
xlm = min(min(0.0003,xlm1),xlm2);

xls1 = xlim(ax1);
xls1 = xls1(2);
xls2 = xlim(ax2);
xls2 = xls2(2);
xls = max(xls1,xls2);

xlim(ax1,[xlm, xls]);
xlim(ax2,[xlm, xls]);

xlm1 = ylim(ax1);
xlm1 = xlm1(1);
xlm2 = ylim(ax2);
xlm2 = xlm2(1);
xlm = min(xlm1,xlm2);


xls1 = ylim(ax1);
xls1 = xls1(2);
xls2 = ylim(ax2);
xls2 = xls2(2);
xls = max(xls1,xls2);

ylim(ax1,[xlm, xls]);
ylim(ax2,[xlm, xls]);


xlabel(ax1,'$\log_{10}(h)$','FontSize',14,'Interpreter','latex');
ylabel(ax1,'$\log_{10}(e)$','FontSize',14,'Interpreter','latex');
annotation('textbox',...
    [0.161937499999999 0.672604238640695 0.0396249999999998 0.0479871495347448],...
    'String','$m=1$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.163239583333332 0.331043729086554 0.0396249999999996 0.0479871495347447],...
    'String','$m=2$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.221833333333331 0.220374939277637 0.0396249999999996 0.0479871495347447],...
    'String','$m=3$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');

xlabel(ax2,'$\log_{10}(h)$','FontSize',14,'Interpreter','latex');
ylabel(ax2,'$\log_{10}(e)$','FontSize',14,'Interpreter','latex');
annotation('textbox',...
    [0.604906249999996 0.659069206793559 0.0396249999999995 0.0479871495347447],...
    'String','$p=1$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.670270833333328 0.2490373596598 0.0417083333333387 0.0479871495347446],...
    'String','$p=2,3$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
