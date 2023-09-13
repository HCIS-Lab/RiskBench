import torch
import torch.nn as nn

__all__ = ['ConvLSTMCell', 'ConvGRUCell']

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias

        def conv2d(in_channels):
            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size-1)//2,
                bias=self.bias,
            )

        self.Wxi = conv2d(self.input_size)
        self.Whi = conv2d(self.hidden_size)
        self.Wxf = conv2d(self.input_size)
        self.Whf = conv2d(self.hidden_size)
        self.Wxg = conv2d(self.input_size)
        self.Whg = conv2d(self.hidden_size)
        self.Wxo = conv2d(self.input_size)
        self.Who = conv2d(self.hidden_size)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cg = torch.tanh(self.Wxg(x) + self.Whg(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        cc = cf * c + ci * cg
        ch = co * torch.tanh(cc)
        return ch, cc

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias

        def conv2d(in_channels):
            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size-1)//2,
                bias=self.bias,
            )

        self.Wxr = conv2d(self.input_size)
        self.Whr = conv2d(self.hidden_size)
        self.Wxz = conv2d(self.input_size)
        self.Whz = conv2d(self.hidden_size)
        self.Wxn = conv2d(self.input_size)
        self.Whn = conv2d(self.hidden_size)

    def forward(self, x, h):
        cr = torch.sigmoid(self.Wxr(x) + self.Whr(h))
        cz = torch.sigmoid(self.Wxz(x) + self.Whz(h))
        cn = torch.tanh(self.Wxn(x) + cr * self.Whn(h))
        ch = (1 - cz) * cn + cz * h
        return ch
