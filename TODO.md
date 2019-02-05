# ToDo

- [ ] Try different optimisers (see [here](https://www.netket.org/docs/optimizers/)):  
  - Adam  
  - AdaMax (less sensitive to learning rate)  
  - AdaGrad (automatically scales LR based on sum over past grads)  
  - RMSProp (scales LR based on exponential moving evg over past grads)  
  - AMSGrad (adjusts learning rate based on "long-term memory" of past grads)  
- [ ] Add noise to training data for curve fitting  
- [x] Implement architecture from Xanadu paper for comparison  

- [ ] Modify QNNBase to be compatible with batch training   
- [ ] Convert curve fitting to use QNNBase   
