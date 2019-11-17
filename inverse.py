
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils
import torch.autograd as autograd

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# **Hyperparameters**

# In[2]:


class Para():
    def __init__(self):
        self.latent_dim = 10 # latent space dimention
        self.img_shape = (1, 28, 28) # image shape (1x28x28 for MNIST)
        self.generator_path = 'D:/OneDrive - California Institute of Technology/Github/PyTorch-GAN/inverse_GD10/G_l10_woBN.pth' # pre-trained generator path
        self.discriminator_path = 'D:/OneDrive - California Institute of Technology/Github/PyTorch-GAN/inverse_GD10/D_l10_woBN.pth' # pre-trained discriminator path
        self.batch_size = 25 # batch size for visualization
        
opt = Para()


# **Network architectures**

# In[3]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(opt.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *opt.img_shape)
        return img

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(opt.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

    
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


# **Load networks**

# In[4]:


generator.load_state_dict(torch.load(opt.generator_path))
discriminator.load_state_dict(torch.load(opt.discriminator_path))


# **Generator and discriminator visualization**

# In[5]:


def vis(y, nrow, title='figure'):
    imgs = torchvision.utils.make_grid(y, nrow=nrow, padding=0, normalize=True)

    plt.figure()
    if cuda:
        plt.imshow(imgs.permute(1, 2, 0).cpu())
    else:
        plt.imshow(imgs.permute(1, 2, 0))
    plt.title(title)
    plt.show()


# In[68]:

# A visualization example

z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))) # latend variables

with torch.no_grad():
    generated_imgs = generator(z) # generate images
    validity = discriminator(generated_imgs) # discriminator validaty

vis(generated_imgs, nrow=int(np.sqrt(opt.batch_size)))

# In[21]:

# GD
# $$ \min_uJ(u)=\frac{1}{2}\|G(u)-y\|_2^2 $$
# $$ u_{k+1}=u_k-\alpha_kDJ(u) $$

u_gt = Variable(Tensor(np.random.normal(0, 1, (5, opt.latent_dim)))) # ground truth of u
with torch.no_grad():
    y = generator(u_gt) # ground truth of y = (u_gt)

# GD
loss = nn.MSELoss()
# start from a different distribution
u_hat = Variable(Tensor(np.random.normal(0, 0.2, (5, opt.latent_dim))), requires_grad=True)
u_0 = u_hat # initial guess
alpha = 5e-3
epoch = 40000

l = []
for i in range(epoch):
    l.append(loss(generator(u_hat), y).item())
    if i % 500 == 0:
        print('['+str(i)+'/'+str(epoch)+']', loss(generator(u_hat), y))
    grad = autograd.grad(outputs=loss(generator(u_hat), y), inputs=u_hat)[0]
    u_hat = u_hat - alpha * grad

plt.plot(l)
plt.title('GD loss curve')
plt.show()

# Visualization of y_gt, y_u0 and y_uhat
vis(y, nrow=5, title='y')
with torch.no_grad():
    y_hat = generator(u_hat)
    y_hat_0 = generator(u_0)
vis(y_hat_0, nrow=5, title='y_hat (before GD)')
vis(y_hat, nrow=5, title='y_hat (after GD)')

# In[84]:

# **Noise model $y=G(u)+\eta$**
# 
# We have prior $u\sim\mathcal(m,I)$ and noise model $\eta\sim\mathcal(0,a\cdot I)$. Using Andrew's notations we will have:
# $$\min_uJ(u)=\frac{1}{2}\|G(u)-y\|_{a\cdot I}^2+\frac{1}{2}\|u-m\|_{I}^2=\frac{1}{2a}\|G(u)-y\|_2^2+\frac{1}{2}\|u-m\|_2^2$$
# $$u_{k+1}=u_k-\alpha_kDJ(u)$$

# generate a single noisy measurement
# GD
# generate several noisy measurements 
# BGD & SGD
# compare their convergence curves
u_gt = np.random.normal(0, 1, (1, opt.latent_dim)) # ground truth of u
with torch.no_grad():
    y = generator(Variable(Tensor(u_gt))) # ground truth of y = (u_gt)

n_measure = 5
u_gt_copy = Variable(Tensor(np.tile(u_gt, (n_measure, 1)))) # copies of ground truth u
a = 0.2 # variance of Gaussian noise
eta = Variable(Tensor(np.random.normal(0, a, (n_measure, *opt.img_shape)))) # noise
with torch.no_grad():
    y_copy = generator(u_gt_copy)
    y_noise = generator(u_gt_copy) + eta # noisy measurement of y from u_gt

# Visualization of y_5 and y_noise
vis(y_copy, nrow=5, title='y_gt (w/o noise)')
vis(y_noise, nrow=5, title='y_measure (w/ Gaussian noise)')

# In[87]:

loss = nn.MSELoss()
m = Tensor(np.zeros((1, opt.latent_dim))) # for computing the regularizer
alpha = 5e-3 # learning rate
epoch = 5000
# initialization
u_hat1 = Variable(Tensor(np.random.normal(0, 0.2, (1, opt.latent_dim))), requires_grad=True)
u_hat2 = u_hat1
u_0 = u_hat1
n_measure = list(y_noise.shape)[0]

l1 = []
l2 = []
for i in range(epoch):
    # compute the objective function
    l1.append((loss(generator(u_hat1), y) + a*loss(u_hat1, m)).item())
    l2.append((loss(generator(u_hat2), y) + a*loss(u_hat2, m)).item())
    if i % 500 == 0:
        print('['+str(i)+'/'+str(epoch)+']', loss(generator(u_hat1), y) + a*loss(u_hat1, m))
        print('['+str(i)+'/'+str(epoch)+']', loss(generator(u_hat2), y) + a*loss(u_hat2, m))
    
    # stochastic gradient descent
    random_idx = np.random.permutation(n_measure)
    for j in range(n_measure):
        grad1 = autograd.grad(outputs=loss(generator(u_hat1), y_noise[random_idx[j],:,:,:]) + a*loss(u_hat1, m), inputs=u_hat1)[0]
        u_hat1 = u_hat1 - alpha * grad1
    # batch gradient descent
    grad2 = 0
    for j in range(n_measure):
        grad2 += autograd.grad(outputs=loss(generator(u_hat2), y_noise[j,:,:,:]) + a*loss(u_hat2, m), inputs=u_hat2)[0]
    u_hat2 = u_hat2 - alpha * grad2 / n_measure

x = np.arange(epoch)
plt.figure()
plt.plot(x, l1, 'g-', x, l2, 'r--')
plt.title('GD loss curve')
plt.legend(('SGD','BGD'))
plt.show()

# Visualization of y_gt, y_u0 and y_uhat
with torch.no_grad():
    y_hat0 = generator(u_0)
    y_hat1 = generator(u_hat1)
    y_hat2 = generator(u_hat2)
vis(torch.cat((y_hat0, y_hat1,y_hat2),0), nrow=5, title='y_initial y_SGD y_BGD')

# In[85]:

# **Noise model $y=G(u)+\eta$**
# 
# We have prior $u\sim\mathcal(m,I)$ and noise model $\eta\sim\mathcal(0,a\cdot I)$. Using Andrew's notations we will have:
# $$\min_uJ(u)=\frac{1}{2}\|G(u)-y\|_{a\cdot I}^2+\frac{1}{2}\|u-m\|_{I}^2=\frac{1}{2a}\|G(u)-y\|_2^2+\frac{1}{2}\|u-m\|_2^2$$
# $$u_{k+1}=u_k-\alpha_kDJ(u)$$

# noise model is Poisson distribution

u_gt = np.random.normal(0, 1, (1, opt.latent_dim)) # ground truth of u
with torch.no_grad():
    y = generator(Variable(Tensor(u_gt))) # ground truth of y = (u_gt)
# range(y)= [-1,1]
    
n_measure = 5
y_num = y.cpu().data.numpy()
y_num = y_num-y_num.min()
y_num = y_num/y_num.max() # convert y to [0,1]
y_noise_temp = np.tile(y_num, (n_measure, 1, 1, 1))
# Simulate low-light noisy images
ppg = 1000 # photons per gray value
for i in range(n_measure):
    temp = y_noise_temp[i,:,:,:]
    p_temp = np.random.poisson(temp * ppg) / ppg # could exceed 1
#    p_temp[p_temp>1] = 1 # convert back to [-1,1]
#    err = np.squeeze(p_temp)-np.squeeze(y.cpu().data.numpy())
#    print(err.max())
#    p_temp = p_temp-p_temp.min()
#    p_temp = p_temp/p_temp.max()
    y_noise_temp[i,:,:,:] = p_temp*2-1+temp

y_noise = Variable(Tensor(y_noise_temp))
y_noise.shape   
# Visualization of y_5 and y_noise
vis(y, nrow=5, title='y_gt (w/o noise)')
vis(y_noise, nrow=5, title='y_measure (w/ Poisson noise)')

# In[87]:


# loss function of Poisson noise model
def Poisson_loss(G_u, y):
    loss = nn.MSELoss()
    # y and G_u range from -1 to 1
    y = y-torch.min(y)
    y = y/torch.max(y) # convert y to [0,1]
    G_u = G_u-torch.min(G_u)
    G_u = G_u/torch.max(G_u) # convert G_u to [0,1]
    return loss(torch.sqrt(G_u), torch.sqrt(y))

loss = nn.MSELoss()
m = Tensor(np.zeros((1, opt.latent_dim))) # for computing the regularizer
alpha = 5e-3 # learning rate
epoch = 5000
# initialization
u_hat1 = Variable(Tensor(np.random.normal(0, 0.2, (1, opt.latent_dim))), requires_grad=True)
u_hat2 = u_hat1
u_0 = u_hat1
n_measure = list(y_noise.shape)[0]
# In[87]:
l1 = []
l2 = []
for i in range(epoch):
    # compute the objective function
    l1.append((Poisson_loss(generator(u_hat1), y) + loss(u_hat1, m)).item())
    l2.append((Poisson_loss(generator(u_hat2), y) + loss(u_hat2, m)).item())
    if i % 500 == 0:
        print('['+str(i)+'/'+str(epoch)+']', Poisson_loss(generator(u_hat1), y) + loss(u_hat1, m))
        print('['+str(i)+'/'+str(epoch)+']', Poisson_loss(generator(u_hat2), y) + loss(u_hat2, m))
    
    # stochastic gradient descent
    random_idx = np.random.permutation(n_measure)
    for j in range(n_measure):
        grad1 = autograd.grad(outputs=Poisson_loss(torch.squeeze(generator(u_hat1)), torch.squeeze(y_noise[random_idx[j],:,:,:])) + loss(u_hat1, m), inputs=u_hat1)[0]
        u_hat1 = u_hat1 - alpha * grad1
    # batch gradient descent
    grad2 = 0
    for j in range(n_measure):
        grad2 += autograd.grad(outputs=Poisson_loss(torch.squeeze(generator(u_hat2)), torch.squeeze(y_noise[j,:,:,:])) + loss(u_hat2, m), inputs=u_hat2)[0]
    u_hat2 = u_hat2 - alpha * grad2 / n_measure

x = np.arange(epoch)
plt.figure()
plt.plot(x, l1, 'g-', x, l2, 'r--')
plt.title('GD loss curve')
plt.show()

# Visualization of y_gt, y_u0 and y_uhat
with torch.no_grad():
    y_hat0 = generator(u_0)
    y_hat1 = generator(u_hat1)
    y_hat2 = generator(u_hat2)
vis(torch.cat((y_hat0, y_hat1,y_hat2),0), nrow=5, title='y_initial y_SGD y_BGD')
