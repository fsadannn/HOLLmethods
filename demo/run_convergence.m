% This script generates the Figures 1-3 and Tables 2-4 of [1] illustrating the convergence rate 
% of the Jacobian-free HOLL scheme JFLLRK for different parameter values. 
%
% [1] Jacobian-free High Order Local Linearization methods for large systems of initial value problems
%     by F.S. Naranjo-Noda and J.C. Jimenez
%


disp('Warning: ');
disp('  If the user does not have license for the Parallel Computing Toolbox,') 
disp('  replace the "parfor" by "for" in the files ')
disp('  convergence_order_test_fj_m_p.m, convergence_order_test_fj_in.m, convergence_order_test_fj_m_p.m')
disp('  ')


abspath = which('run_convergence');
pos = strfind(abspath, filesep); pos = pos(end);
abspath = abspath(1:pos - 1);

cd(sprintf('%s%s%s',abspath,filesep,'..'));

s = [
 [abspath,filesep,'..',filesep,'demo;']...,
 [abspath,filesep,'..',filesep,'llint;']...,
 [abspath,filesep,'Brusselator;']
];

path(s,path);
initllPaths(true);

convergence_order_test_fj_m_p;
convergence_order_test_fj_in;
convergence_order_test_fj_out;