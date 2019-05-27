import torch
import matplotlib.pyplot as plt
import mnist_classifier
from matplotlib.widgets import Slider

def main():
    model = mnist_classifier.Net()
    model.load_state_dict(torch.load("data/mnist_cnn.pt"))
    model.eval()
    tmp = torch.load("./data/x_y_train.pt")
    y = tmp[1]
    x = tmp[0]
    del(tmp)

    print("sample: ground truth", y[0],
            "\nmodel predictions: ", model(x[0:1, ...]).exp().detach().numpy())

    #let's find the gradient of the p to be 5, wrt the input.
    sample = x[0:1, ...] #keep first dimension with 1 element around cause that's what the model needs
    sample.requires_grad=True
    output = model(sample).exp()[..., 5]
    print("output:", output)
    output.backward()
    print("L2 norm of gradient", sample.grad.norm())
    print("maximum of gradient", sample.grad.abs().max())

    scale=30
    perturbation = -scale*sample.grad
    adv_sample = perturb(sample, perturbation)
    adv_output = model(adv_sample).exp()
    print("output on adv sample:", adv_output)

    #visualize
    plt.ion()
    fig = plt.figure()
    #hidpi workaround
    fig.dpi = 2*fig.dpi
    ax_original_img = fig.add_subplot(231)
    ax_perturbation_img = fig.add_subplot(232)
    ax_adversarial_img = fig.add_subplot(233)
    ax_original_predictions = fig.add_subplot(234)
    ax_adversarial_predictions = fig.add_subplot(236)

    ax_slider = fig.add_subplot(235)
    slider_scale = Slider(ax_slider, 'scale', 0, 100., valinit = 0.) 

    adv_sample = perturb(sample, perturbation)
    original_img = sample.detach()[0,0,:,:]
    ax_original_img.imshow(original_img, cmap='gray')
    ax_perturbation_img.imshow(perturbation.detach()[0,0,:,:], cmap='gray')
    adv_imshow = ax_adversarial_img.imshow(adv_sample.detach()[0,0,:,:], cmap='gray')
    ax_original_predictions.bar(range(10), model(sample).exp().detach()[0,:])
    ax_original_predictions.set_ylim(0., 1.)
    adv_bar_rects = ax_adversarial_predictions.bar(range(10), model(adv_sample).exp().detach()[0,:])
    ax_adversarial_predictions.set_ylim(0., 1.)

    def update(val):
        scale = slider_scale.val
        perturbation = -scale*sample.grad
        adv_sample = perturb(sample, perturbation)
        adv_imshow = ax_adversarial_img.imshow(adv_sample.detach()[0,0,:,:], cmap='gray')
        adv_predictions = model(adv_sample).exp().detach()[0, :]
        [rect.set_height(h) for rect, h in zip(adv_bar_rects, adv_predictions)]
        fig.canvas.draw()


    slider_scale.on_changed(update)





    input("press enter...")

def perturb(image, perturbation):
    #makes sure image+perturbation keeps the same range, which is -0.4242 until 2.8215.
    out = image + perturbation
    out = out.clamp(-.4242, 2.8215)
    return out

if __name__=='__main__':
    main()
