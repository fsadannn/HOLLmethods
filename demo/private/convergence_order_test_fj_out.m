clear all
% close all
N = 50;

fname = @f_brusselator;
Jname = @J_brusselator;
x0= [1+sin((2*pi/(N+1))*(1:N)),repmat(3,1,N)];
IT = [0,1];

options15s = odeset('RelTol',1.0e-12,'AbsTol',1.0e-14,'Jacobian',Jname);

kdim = 8;
pade = 2;
order_out = 1;
order_in = 2;
proportion_out = 0;
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

interval = 2.^(-(7:10));
e4 = zeros(1,length(interval));
proportion_out = [0,0.5,1,1.5];

Result4L.eta2=[1; 1; 1; 1];
Result4L.alpha2=[0; 0.5; 1; 1.5];
Result4L.r=[2; 2.5; 3; 3.5];
Result4L.est_r=[];
Result4L.ci=[];
for j=1:length(proportion_out)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out(j),proportion_in,...
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

    Result4L.est_r=[Result4L.est_r; c4(1)];
    Result4L.ci=[Result4L.ci; confidence4];
end
disp(' ')
disp('Table 4, left ')
Table4L = struct2table(Result4L);
disp(Table4L);
    
order_out = 2;
proportion_out = [0,1/8,0.25];
interval = 2.^(-(6:9));

Result4R.eta2=[2; 2; 2; 2];
Result4R.alpha2=[0; 0.125; 0.25; 0.5];
Result4R.r=[3; 3.25; 3.5; 4];
Result4R.est_r=[];
Result4R.ci=[];
for j=1:length(proportion_out)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out(j),proportion_in,...
             proportion_in_sub);

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
    
    Result4R.est_r=[Result4R.est_r; c4(1)];
    Result4R.ci=[Result4R.ci; confidence4];
end

order_out = 2;
proportion_out = [0.5];
interval = 2.^(-(5:8));

for j=1:length(proportion_out)
    parfor i=1:length(interval)
        t = IT(1):interval(i):IT(2);

         y4 = JFLLRK(fname,t,x0,1.0e-9,1.0e-12,kdim,pade,...
             order_out,order_in,proportion_out(j),proportion_in,...
             proportion_in_sub);

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
    
    Result4R.est_r=[Result4R.est_r; c4(1)];
    Result4R.ci=[Result4R.ci; confidence4];
end
disp(' ')
disp('Table 4, right ')
Table4R = struct2table(Result4R);
disp(Table4R);


xlm1 = xlim(ax1);
xlm1 = xlm1(1);
xlm2 = xlim(ax2);
xlm2 = xlm2(1);
xlm = min(min(0.0008,xlm1),xlm2);

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


title(ax1,'$\eta_1=1$','FontSize',14,'Interpreter','latex');
xlabel(ax1,'$\log_{10}(h)$','FontSize',14,'Interpreter','latex');
ylabel(ax1,'$\log_{10}(e)$','FontSize',14,'Interpreter','latex');

annotation('textbox',...
    [0.141624999999999 0.666575059044638 0.0396249999999997 0.0479871495347448],...
    'String','$\alpha_2=0$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.138109374999999 0.506481227755152 0.0450937500000008 0.0479871495347448],...
    'String','$\alpha_2=0.5$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.143187499999999 0.337128137206409 0.0396249999999997 0.0479871495347449],...
    'String','$\alpha_2=1$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.139932291666665 0.192307692307692 0.0423593750000013 0.0388995057427644],...
    'String','$\alpha_2=1.5$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');

title(ax2,'$\eta_1=2$','FontSize',14,'Interpreter','latex');
xlabel(ax2,'$\log_{10}(h)$','FontSize',14,'Interpreter','latex');
ylabel(ax2,'$\log_{10}(e)$','FontSize',14,'Interpreter','latex');
annotation('textbox',...
    [0.603473958333326 0.375 0.0396249999999999 0.0369446872118521],...
    'String','$\alpha_2=0$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.586416666666658 0.284630614600181 0.0559010416666735 0.0479871495347447],...
    'String','$\alpha_2=0.125$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.592536458333326 0.206104973574543 0.0509531250000067 0.0479871495347451],...
    'String','$\alpha_2=0.25$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');
annotation('textbox',...
    [0.657640624999992 0.176858178702747 0.0436614583333413 0.0479871495347443],...
    'String','$\alpha_2=0.5$',...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FontSize',12,...
    'FitBoxToText','off');