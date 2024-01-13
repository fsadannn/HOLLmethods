# High Order Local Linearization methods

## This repo hold the original source code used for the Local Linearization papers

This Matlab toolbox provides the <strong>Jacobian-free High Order Local Linearization (HOLL)</strong> schemes JFLLRK and JFLLRK4 described in [1] for the integration of large systems of initial value problems.

[1] Jacobian-free High Order Local Linearization methods for large systems of initial value problems
    by F.S. Naranjo-Noda and J.C. Jimenez
## <strong>Jacobian-free HOLL</strong>

### [```JFLLRK4```](./llint/JFLLRK4.m) fixed step-size Jacobian-free Locally Linearized Runge-Kuttta scheme of order 4

### [```JFLLRK```](./llint/JFLLRK4.m) fixed step-size Jacobian-free Locally Linearized Runge-Kuttta scheme with variable parameters

<br/>

### <strong>Demos</strong>

[`run_examples.m`](./demo/run_examples.m) generates the Tables 5-8 of [1] illustrating the performance of the Jacobian-free HOLL scheme JFLLRK4 in the integration of test examples.

[`run_convergence.m`](./demo/run_convergence.m) generates the Figures 1-3 and Tables 2-4 of [1] illustrating the convergence rate of the Jacobian-free HOLL scheme JFLLRK for different parameter values.

