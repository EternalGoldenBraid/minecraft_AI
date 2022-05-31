import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_states, outputs, hid_dim):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(n_states,hid_dim)
        self.l2 = nn.Linear(hid_dim,hid_dim)
        self.l3 = nn.Linear(hid_dim, outputs)


    def forward(self, x):

        x = x.to(device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    #def __init__(self,h,w, outputs):
        # For vision version.
        #self.conv1 = nn.conv2d(500,12,kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(12)
        #self.conv1 = nn.conv2d(12,6,kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(12)

        # Compute size of outputs of conv2d
        # CONFIRM THE MATH? WHAT AND WHY?
        #def conv2d_size_out(size, kernel_size=5, stride =2):
        #    return (size- (kernel_size-1) -1)//stride + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw*convh*32
        #self.head = nn.Linear(linear_input_size, outputs)
