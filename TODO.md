# ToDo

- [ ] Try different optimisers (see [here](https://www.netket.org/docs/optimizers/)):  
  - [x] Adam  
  - [ ] AdaMax (less sensitive to learning rate)  
  - [ ] AdaGrad (automatically scales LR based on sum over past grads)  
  - [ ] RMSProp (scales LR based on exponential moving evg over past grads)  
  - [ ] AMSGrad (adjusts learning rate based on "long-term memory" of past grads)  
- [x] Add noise to training data for curve fitting  
- [x] Implement architecture from Xanadu paper for comparison  
- [ ] Generate random set of initial params and load them at startup

- [x] Modify QNNBase to be compatible with batch training   
- [x] Convert curve fitting to use QNNBase   
- [x] Fix StateEngineer so it's compatible with batched version of QNNBase
- [x] Add save function to QNNBase
