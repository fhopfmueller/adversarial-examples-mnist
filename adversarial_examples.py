import torch
import matplotlib.pyplot as plt
import mnist_classifier

def main():
    model = mnist_classifier.Net()
    model.load_state_dict(torch.load("data/mnist_cnn.pt"))
    model.eval()
    tmp = torch.load("./data/x_y_train.pt")
    y = tmp[1]
    x = tmp[0]
    del(tmp)

    print("first sample: ground truth", y[0],
    "\nmodel says ", model(x[0:1, ...]).exp())

    #let's find the gradient of the p to be 5, wrt the input.
    sample = x[0:1, ...]
    sample.requires_grad=True
    output = model(sample).exp()[..., 5]
    print("output:", output)
    output.backward()
    print("shape of gradient at sample:", sample.grad.shape)
    print("L2 norm of gradient", sample.grad.norm())
    print("maximum of gradient", sample.grad.abs().max())
    print("range of input", x.min(), x.max())

    scale=30
    #the gradients for the p to be anything:
    sample.grad.zero_()
    output = model(sample).exp()[..., 3]
    print("output:", output)
    output.backward()
    print("shape of gradient at sample:", sample.grad.shape)
    print("L2 norms of gradient", sample.grad.norm())
    print("maximum of gradient", sample.grad.abs().max())

    adv_sample = sample + scale*sample.grad
    adv_output = model(adv_sample).exp()
    print("output on adv sample:", adv_output)

    #visualize
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(sample.detach()[0,0,:,:], cmap='gray')
    ax2.imshow(sample.grad.detach()[0,0,:,:], cmap='gray')
    ax3.imshow(adv_sample.detach()[0,0,:,:], cmap='gray')

    plt.show()


def probs(tensor):
    return model(tensor).exp()

#not working...
def constrained_gd( f, parameters, constraint, lr=.1, its=100 ):
    # function: a scalar function with argument of type parameters
    # parameters: a torch tensor
    # constraints: a function of parameters which will be enforced to be \le 0
    # returns: the locus x of the minimum of function, the lagrange multiplier (if it's zero, the constraint is saturated)
    x = parameters.clone()
    x.requires_grad = True
    lm = torch.Tensor([1.])
    lm.requires_grad = True
    #lagrangian.backward(retain_graph=True)

    for i in range(its):
        lagrangian = f(x) - lm * lm * constraint(x)
        lagrangian.backward()
        print(lagrangian, x, lm)
        with torch.no_grad():
            x = x - lr * x.grad
            lm = lm - lr * lm.grad
        x.requires_grad=True
        lm.requires_grad=True
    return x,lm

if __name__=='__main__':
    main()
    #def f(t):
    #    return t[0]*t[1]
    #def c(t):
    #    return t[0]*t[0] + t[1]*t[1] - 1.

    #t = torch.Tensor( [1.5, 1.])
    #print(t)
    #print(f(t))
    #print(c(t))
    #x,_=constrained_gd(f, t, c)
    #print(c(x))

