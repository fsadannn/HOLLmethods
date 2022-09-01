clear all
% close all
N = 50;

fname = @f_brusselator;
Jname = @J_brusselator;
x0= [1+sin((2*pi/(N+1))*(1:N)),repmat(3,1,N)];
IT = [0,1];

options15s = odeset('RelTol',1.0e-12,'AbsTol',1.0e-14,'Jacobian',Jname);

kdim = 14;
pade = 4;
order_out = 2;
order_in = 1;
proportion_out = 1;
proportion_in = 1;
proportion_in_sub = 1;
algo = 2;



figure
ax1 = subplot(1,2,1,'XScale', 'log', 'YScale', 'log');
ax2 = subplot(1,2,2,'XScale', 'log', 'YScale', 'log');
hold(ax1,'on')
hold(ax2,'on')
box(ax1,'on')
box(ax2,'on')

interval = 2.^(-(6:9));
e4 = zeros(1,length(interval));
proportion_in = [0,1];
proportion_in_sub = 1;

Result3L.eta1=[1; 1; 1];
Result3L.beta=[1; 1; 1];
Result3L.alpha1=[0; 1; 2];
Result3L.r=[2; 3; 4];
Result3L.est_r=[];
Result3L.ci=[];
for j=1:length(proportion_in)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out,proportion_in(j),...
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
    
    Result3L.est_r=[Result3L.est_r; c4(1)];
    Result3L.ci=[Result3L.ci; confidence4];
end


proportion_in = 2;
proportion_in_sub = [1];
interval = 2.^(-(5:8));

for j=1:length(proportion_in_sub)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out,proportion_in,...
             proportion_in_sub(j));

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

    Result3L.est_r=[Result3L.est_r; c4(1)];
    Result3L.ci=[Result3L.ci; confidence4];
end
disp(' ')
disp('Table 3, left ')
Table3L = struct2table(Result3L);
disp(Table3L);


% 
order_in = 2;
proportion_in = [0.5,1];
proportion_in_sub = 0;
interval = 2.^(-(6:9));

Result3R.eta1=[2; 2; 2];
Result3R.beta=[0; 0; 0];
Result3R.alpha1=[0.5; 1; 1.5];
Result3R.r=[2; 3; 4];
Result3R.est_r=[];
Result3R.ci=[];
for j=1:length(proportion_in)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out,proportion_in(j),...
             proportion_in_sub);

        [~,sol] = ode15s(fname,t,x0,options15s);

        e4(i)=AbsErr(sol,y4);
    end


    linterval4 = log10(interval);
    le4 = log10(e4);
    [c4,S4] = polyfit(linterval4,le4,1);
    pe4 = polyval(c4,linterval4,S4);

    iR4 = S4.R\eye(2);    cov4 = (iR4*iR4').*(S4.normr^2/S4.df);

    plot(ax2,10.^linterval4,10.^le4, '--o',10.^linterval4, 10.^pe4, '-' );
    % 90 percentil
    percentilev4=tinv(0.95,S4.df-1);
    
    confidence4 = percentilev4*sqrt(cov4(1,1)/(S4.df));
    
    Result3R.est_r=[Result3R.est_r; c4(1)];
    Result3R.ci=[Result3R.ci; confidence4];

end
 
% 
order_in = 2;
interval = 2.^(-(5:8));
proportion_in = 0;
proportion_in_sub = [1.5];

for j=1:length(proportion_in_sub)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out,proportion_in,...
             proportion_in_sub(j));

        [~,sol] = ode15s(fname,t,x0,options15s);

        e4(i)=AbsErr(sol,y4);
    end


    linterval4 = log10(interval);
    le4 = log10(e4);
    [c4,S4] = polyfit(linterval4,le4,1);
    pe4 = polyval(c4,linterval4,S4);

    iR4 = S4.R\eye(2);    cov4 = (iR4*iR4').*(S4.normr^2/S4.df);

    plot(ax2,10.^linterval4, 10.^le4, '--o',10.^linterval4, 10.^pe4, '-' );
    % 90 percentil
    percentilev4=tinv(0.95,S4.df-1);
    
    confidence4 = percentilev4*sqrt(cov4(1,1)/(S4.df));
    
    Result3R.est_r=[Result3R.est_r; c4(1)];
    Result3R.ci=[Result3R.ci; confidence4];
end
disp(' ')
disp('Table 3, right ')
Table3R = struct2table(Result3R);
disp(Table3R);


xlm1 = xlim(ax1);
xlm1 = xlm1(1);
xlm2 = xlim(ax2);
xlm2 = xlm2(1);
xlm = min(xlm1,xlm2);

xls1 = xlim(ax1);
xls1 = xls1(2);
xls2 = xlim(ax2);
xls2 = xls2(2);
xls = max(xls1,max(0.04,xls2));

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


title(ax1,'$\eta_1=1, \beta=1$','FontSize',14,'Interpreter','latex');
xlabel(ax1,'$\log_{10}(h)$','FontSize',14,'Interpreter','latex');
ylabel(ax1,'$\log_{10}(e)$','FontSize',14,'Interpreter','latex');
annotation('textbox',...
    [0.164541666666666 0.720644178331149 0.0396249999999998 0.0479871495347448],...
    'String','$\alpha_1=0$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.167015624999999 0.40008952906524 0.0396249999999997 0.047987149534745],...
    'String','$\alpha_1=1$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.206859374999999 0.278556086977147 0.0396249999999997 0.047987149534745],...
    'String','$\alpha_1=2$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');


title(ax2,'$\eta_1=2, \beta=0$','FontSize',14,'Interpreter','latex');
xlabel(ax2,'$\log_{10}(h)$','FontSize',14,'Interpreter','latex');
ylabel(ax2,'$\log_{10}(e)$','FontSize',14,'Interpreter','latex');
annotation('textbox',...
    [0.602822916666663 0.55289701463585 0.0430104166666698 0.0479871495347449],...
    'String','$\alpha_1=0.5$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.608812499999996 0.25873169859047 0.0430104166666698 0.0479871495347449],...
    'String','$\alpha_1=1$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.682510416666662 0.199574486272804 0.0430104166666698 0.0479871495347451],...
    'String','$\alpha_1=1.5$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
