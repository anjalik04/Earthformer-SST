import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM Cell implementation.
    Based on the implementation from https://github.com/ndrplz/ConvLSTM_pytorch
   .[14, 18]
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        # This single convolution layer computes all 4 gates (i, f, o, g) at once
        # for efficiency.
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass for a single timestep.

        Parameters
        ----------
        input_tensor: (B, C_in, H, W)
            Input tensor for the current timestep.
        cur_state: (tuple)
            Tuple containing the previous hidden state (h_cur) and cell state (c_cur).
            h_cur: (B, C_hidden, H, W)
            c_cur: (B, C_hidden, H, W)

        Returns
        -------
        h_next, c_next: (tuple)
            Tuple containing the next hidden state and cell state.
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, C_in + C_hidden, H, W)

        # Compute all 4 gates
        combined_conv = self.conv(combined)
        
        # Split the 4 gates (i, f, o, g)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate (new candidate)

        # Compute next cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initializes the hidden and cell states to zeros.
        
        Parameters
        ----------
        batch_size: int
            Size of the batch.
        image_size: (int, int)
            Height and width of the spatial dimensions.
        
        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Zero-initialized hidden state and cell state.
        """
        height, width = image_size
        # Use self.conv.weight.device to ensure states are on the same device as the model
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    A multi-layer ConvLSTM implementation that wraps ConvLSTMCell.
    Based on the implementation from https://github.com/ndrplz/ConvLSTM_pytorch
   .[14, 18]

    Input must be 5D: (B, T, C, H, W) or (T, B, C, H, W)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        """
        Initialize the multi-layer ConvLSTM.

        Parameters
        ----------
        input_dim: int
            Number of channels in input tensor.
        hidden_dim: list of int
            List of hidden channels for each layer.
        kernel_size: list or tuple of (int, int)
            List of kernel sizes for each layer.
        num_layers: int
            Number of ConvLSTM layers.
        batch_first: bool
            Whether the 1st dimension is batch (B, T, C, H, W).
        bias: bool
            Whether or not to add bias in convolutions.
        return_all_layers: bool
            If True, return list of (layer_output, last_state) for all layers.
            If False, return (last_layer_output, last_layer_last_state).
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure hidden_dim and kernel_size are lists for iteration
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for the entire sequence.

        Parameters
        ----------
        input_tensor: torch.Tensor
            5D Tensor of shape (B, T, C, H, W) or (T, B, C, H, W)
        hidden_state: list of (h, c) tuples, optional
            Initial hidden state for each layer. If None, initialized to zeros.

        Returns
        -------
        layer_output_list, last_state_list
            layer_output_list: List of 5D Tensors.
            last_state_list: List of (h, c) tuples.
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Iterate over time dimension
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                  cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1) # (B, T, C_hidden, H, W)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # layer_output_list is (B, T, C_hidden_last, H, W)
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
