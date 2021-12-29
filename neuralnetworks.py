import numpy as np
import optimizers as opt
import mlfuncs
import sys  # for sys.float_info.epsilon

######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():

    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):
        '''
        n_inputs: int
        n_hidden_units_by_layers: list of ints, or empty
        n_outputs: int
        '''

        self.n_inputs = n_inputs
        self.n_hidden_units_by_layers = n_hidden_units_by_layers
        self.n_outputs = n_outputs

        # Build list of shapes for weight matrices in each layera
        shapes = []
        n_in = n_inputs
        for nu in self.n_hidden_units_by_layers + [n_outputs]:
            shapes.append((n_in + 1, nu))
            n_in = nu

        self.all_weights, self.Ws = self._make_weights_and_views(shapes)
        self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

        self.total_epochs = 0
        self.error_trace = []
        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

        self.kind = 'regression'

    def _make_weights_and_views(self, shapes):
        '''
        shapes: list of pairs of ints for number of rows and columns
                in each layer
        Returns vector of all weights, and views into this vector
                for each layer
        '''
        all_weights = np.hstack([np.random.uniform(size=shape).flat
                                 / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements
        # from vector of all weights into correct shape for each layer.
        views = []
        first_element = 0
        for shape in shapes:
            n_elements = shape[0] * shape[1]
            last_element = first_element + n_elements
            views.append(all_weights[first_element:last_element]
                         .reshape(shape))
            first_element = last_element

        return all_weights, views

    def __repr__(self):
        return f'{type(self).__name__}({self.n_inputs}, ' + \
            f'{self.n_hidden_units_by_layers}, {self.n_outputs})'

    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final objective value is {self.error_trace[-1]:.4g}.'
        return s
 
    def _standardize(self, X, T=None):
        if self.X_means is None:
            self.X_means = X.mean(axis=0)
            self.X_stds = X.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1
            if T is not None:
                self.T_means = T.mean(axis=0)
                self.T_stds = T.std(axis=0)

        # Standardize X and T
        X = (X - self.X_means) / self.X_stds
        if T is not None:
            T = (T - self.T_means) / self.T_stds

        if T is not None:
            return X, T
        else:
            return X
            
    def _unstandardize_T(self, T):
        if self.T_means is not None:
            return T * self.T_stds + self.T_means
        else:
            return T

    # Only used for classification networks
    def _make_indicator_vars(self, T):
        '''Assumes argument is N x 1, N samples each being integer class label.'''
        # Make sure T is two-dimensional. Should be nSamples x 1.
        if T.ndim == 1:
            T = T.reshape((-1, 1))    
        return (T == self.classes).astype(float)  # to work on GPU

    def _error_convert(self, err):
        sqrt_err = np.sqrt(err) * self.T_stds
        return sqrt_err[0] if isinstance(sqrt_err, np.ndarray) else sqrt_err 

    def make_batches(X, T, batch_size=None):
        if batch_size is None:
            yield X, T

    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, momentum=0, batch_size=None, verbose=True):
        '''
        X: n_samples x n_inputs matrix of input samples, one per row
        T: n_samples x n_outputs matrix of target output values,
            one sample per row
        n_epochs: number of passes to take through all samples
            updating weights each pass
        method: 'sgd', 'adam', or 'scg'
        learning_rate: factor controlling the step size of each update
        '''

        if self.kind == 'regression':
            X, T = self._standardize(X, T)
        else:
            X = self._standardize(X)
            T = self._make_indicator_vars(T)
        
        # Instantiate Optimizers object by giving it vector of all weights

        self.error_trace = []

        if method == 'sgd':
            optimizer = opt.SGD(self.all_weights)
        elif method == 'adam':
            optimizer = opt.Adam(self.all_weights)
        # elif method == 'scg':
        #     print('SCG needs work!!!!')
        #     optimizer = opt.SCG(self.all_weights)
        else:
            raise Exception("method must be 'sgd', or 'adam')  # , or 'scg'")

        print_every = n_epochs // 10
        if print_every == 0:
            print_every = 1

        for epoch in range(n_epochs):

            batches = mlfuncs.make_batches(X, T, batch_size)
            n_batches = 0
            for Xbatch, Tbatch in batches:

                # print(f'Training {epoch=} {batchi=}')
                error = optimizer.step(self._error_f, self._gradient_f, fargs=[Xbatch, Tbatch],
                                       learning_rate=learning_rate, momentum=momentum)
                n_batches += 1

            self.error_trace.append(self._error_convert(error))

            if verbose and ((epoch + 1) == n_epochs or (epoch + 1) % print_every == 0):
                print(f'{method}: Epoch {epoch+1:d} {n_batches=:d} ObjectiveF={self.error_trace[-1]:.5f}')

        self.total_epochs += len(self.error_trace)

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self

    def _forward(self, X):
        '''
        X assumed to be standardized
        '''
        self.Ys = [X]
        for W in self.Ws[:-1]:  # forward through all but last layer
            self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def _error_f(self, X, T):
        Ys = self._forward(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def _gradient_f(self, X, T):
        # Assumes forward_pass just called with layer outputs saved in self.Ys.
        n_samples = X.shape[0]
        n_outputs = T.shape[1]

        # D is delta matrix to be back propagated
        D = -(T - self.Ys[-1]) / (n_samples * n_outputs)
        self._backpropagate(D)

        return self.all_gradients

    def _backpropagate(self, D):
        # Step backwards through the layers to back-propagate the error (D)
        n_layers = len(self.n_hidden_units_by_layers) + 1
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.Grads[layeri][1:, :] = self.Ys[layeri].T @ D
            # gradient of just the bias weights
            self.Grads[layeri][0:1, :] = np.sum(D, axis=0)
            # Back-propagate this layer's delta to previous layer
            if layeri > 0:
                D = D @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = self._standardize(X)
        Ys = self._forward(X)
        # Unstandardize output Y before returning it
        return self._unstandardize_T(Ys[-1])

    def get_error_trace(self):
        return self.error_trace


######################################################################
## class NeuralNetworkClassifier(NeuralNetwork)
######################################################################

class NeuralNetworkClassifier(NeuralNetwork):

    def __init__(self, n_inputs, n_hidden_units_by_layers, classes):
        '''
        n_inputs: int
        n_hidden_units_by_layers: list of ints, or empty
        classes: list of all unique class labels
        '''
        # wrap self.classes in np.array in case classes given as a list to the constructor
        self.classes = np.array(classes).reshape(-1)
        n_outputs = len(classes)
        super(NeuralNetworkClassifier, self).__init__(n_inputs, n_hidden_units_by_layers, n_outputs)
        self.kind = 'classification'

    def _softmax(self, Y):
        '''Apply to final layer weighted sum outputs'''
        # Trick to avoid overflow
        # maxY = max(0, self.max(Y))
        maxY = Y.max()  #self.max(Y))        
        expY = np.exp(Y - maxY)
        denom = expY.sum(1).reshape((-1, 1))
        Y = expY / (denom + sys.float_info.epsilon)
        return Y

    def _error_convert(self, err):
        return np.exp(-err)

    # Function to be minimized by optimizer method, mean squared error
    def _neg_log_likelihood_f(self, X, T):
        Ys = self._forward(X)
        Y = self._softmax(Ys[-1])
        neg_mean_log_likelihood = -np.mean(T * np.log(Y + sys.float_info.epsilon))
        return neg_mean_log_likelihood

    # # train() calls _error_f so must define it here
    def _error_f(self, X, T):
        return self._neg_log_likelihood_f(X, T)

    # Gradient of function to be minimized for use by optimizer method
    def _gradient_f(self, X, T):
        # Assumes forward_pass just called with layer outputs saved in self.Ys.
        n_samples = X.shape[0]
        n_outputs = T.shape[1]

        Y = self._softmax(self.Ys[-1])

        # D is delta matrix to be back propagated
        D = -(T - Y) / (n_samples * n_outputs)
        self._backpropagate(D)

        return self.all_gradients

    def use(self, X):
        '''X assumed to not be standardized. Returns (classes, class_probabilities)'''
        # Standardize X
        X = self._standardize(X)
        Ys = self._forward(X)
        Y = self._softmax(Ys[-1])
        classes = self.classes[np.argmax(Y, axis=1)].reshape(-1, 1)
        return classes, Y



######################################################################
## class NeuralNetworkClassifier_CNN(NeuralNetworkClassifier)
######################################################################

class NeuralNetworkClassifier_CNN(NeuralNetworkClassifier):

    def __init__(self, n_inputs, conv_layers, fc_layers, classes):
        '''
        n_inputs: image size is n_inputs x n_inputs x n_channels
        conv_layers: list of lists of ints, each being n_units, kernel_size, stride
        fc_layers: list of n_units per fully-connected layers
        classes: list of unique class labels
        '''

        self.n_inputs = n_inputs
        # wrap self.classes in np.array in case classes given as a list to the constructor
        self.classes = np.array(classes).reshape(-1)
        self.n_outputs = len(classes)
        self.layers = [{'n_units': nu, 'kernel': k, 'stride': s} for (nu, k, s) in conv_layers] + \
            [{'n_units': nu} for nu in fc_layers] + [{'n_units': self.n_outputs, 'final': True}]

        # Build list of shapes for weight matrices in each layera
        shapes = []
        in_rc, in_rc, n_channels = n_inputs
        
        prev_layer = None
        for layer in self.layers:
            if 'kernel' in layer:
                n_in = 1 + layer['kernel'] ** 2 * n_channels
                W_shape = (n_in, layer['n_units'])
                out_rc = (in_rc - layer['kernel']) // layer['stride'] + 1
                if out_rc < 1:
                    raise Exception(f'Layer {layer=} cannot be created for the input size of {in_rc=}')
                layer.update({'in_channels': n_channels,
                              'in_rc': in_rc,
                              'out_rc': out_rc})
                shapes.append(W_shape)
                n_channels = layer['n_units']
                in_rc = layer['out_rc']
                prev_layer = layer
            else:
                if not prev_layer:
                    n_in = 1 + in_rc ** 2 * n_channels
                elif 'kernel' in prev_layer:
                    n_in = 1 + in_rc ** 2 * prev_layer['n_units']
                nu = layer['n_units']
                shapes.append((n_in, nu))
                n_in = 1 + nu
                prev_layer = layer
                
        self.all_weights, self.Ws = self._make_weights_and_views(shapes)
        self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

        # Store references to W and G in layer dictionaries
        for layer, W, G in zip(self.layers, self.Ws, self.Grads):
            layer['W'] = W
            layer['G'] = G

        self.total_epochs = 0
        self.error_trace = []
        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

        self.kind = 'classification'

    def __repr__(self):
        conv_layers = [(lay['n_units'], lay['kernel'], lay['stride']) for lay in self.layers if 'kernel' in lay]
        fc_layers = [lay['n_units'] for lay in self.layers if 'kernel' not in lay]
        details = ''
        for i, layer in enumerate(self.layers):
            if 'kernel' in layer:
                details += (f"\n Layer {i}: n_units={layer['n_units']} "
                            f"kernel={layer['kernel']} "
                            f"stride={layer['stride']} "
                            f"in_channels={layer['in_channels']} "
                            f"in_rc={layer['in_rc']} "
                            f"out_rc={layer['out_rc']}")
            else:
                if i != len(self.layers) - 1:
                    details += f"\n Layer {i}: n_units={layer['n_units']}"
                else:
                    details += ''
        return f'{type(self).__name__}({self.n_inputs}, ' + \
            f'{conv_layers}, {fc_layers}, {self.classes})' + details
                

    def _forward(self, X):
        '''X assumed to be standardized
        '''
        debug = False
        
        N = X.shape[0]
        self.Ys = [X]
        n_layers = len(self.layers)
        for layer in self.layers:
            if debug: print(f'     ======== Input {self.Ys[-1].shape=}')
            W = layer['W']
            if 'kernel' in layer:
                Y = self._convolve(self.Ys[-1], layer)
                Y = np.tanh(Y + W[np.newaxis, np.newaxis, np.newaxis, 0, :])
            else:
                Y = self.Ys[-1]
                if Y.ndim > 2:
                    Y = Y.reshape(N, -1)
                Y = Y @ W[1:, :] + W[0:1, :]
                if 'final' not in layer:
                    Y = np.tanh(Y)
            self.Ys.append(Y)
            if debug: print(f'     ======== Output {self.Ys[-1].shape=}')

        return self.Ys

    def _backpropagate(self, Delta):
        # Step backwards through the convolutional layers to back-propagate the error (D)
        # Assumes self.Ys from convolutional layers represented as patches, as computed by _forward

        debug = False
        
        for layeri in range(len(self.layers) - 1, -1, -1):
            if debug: print(f'{layeri=} {Delta.shape=}')
            if layeri < len(self.layers) - 1:
                outs = self.Ys[layeri + 1]
                ignore_input_patches = outs.shape[1] - Delta.shape[1]
                if ignore_input_patches > 0:
                    Delta *= 1 - self.Ys[layeri + 1] ** 2
            layer = self.layers[layeri]
            W = layer['W']
            if 'kernel' in layer:  # Convolutional layer
                Yin_patches = self._make_patches(self.Ys[layeri], layer['kernel'], layer['stride'])
                N, D, D, K, K, Uprev = Yin_patches.shape
                N, D, D, U = Delta.shape
                G = (Yin_patches.reshape(N * D * D, K * K * Uprev).T @ Delta.reshape(N * D * D, U))
                layer['G'][1:, :] = G
                layer['G'][0:1, :] = np.sum(Delta, axis=tuple(range(Delta.ndim - 1)))
                if layeri > 0:
                     # Delta = self._convolve_backprop(Delta, W[1:, :], layer['kernel'], layer['stride'])
                    # prev_layer = self.layers[layeri - 1]
                    Delta = self._convolve_backprop(Delta, W[1:, :], layer['kernel'], layer['stride'])
                if debug: print(f'Backpropagating {Delta.shape=}')
                # print(f'{layeri=} {Delta.shape=} {self.Ys[layeri+1].shape=}')
            else:
                # Fully connected
                N = Delta.shape[0]
                layer['G'][1:, :] = self.Ys[layeri].reshape(N, -1).T @ Delta
                layer['G'][0:1, :] = np.sum(Delta, axis=0)
                Delta = Delta @ W[1:, :].T
                Delta = Delta.reshape(self.Ys[layeri].shape)
                
        return self.all_gradients

    def _make_patches(self, X, patch_size, stride=1):
        '''
        X: n_samples x n_rows x n_cols x n_channels (r_rows == n_cols)
        patch_size: number of rows (= number of columns) in each patch
        stride: number of pixels to shfit to next patch (n rows = n columns)
        '''
        X = np.ascontiguousarray(X)  # make sure X values are contiguous in memory

        # print(f'make_patches: {X.shape=} {patch_size=} {stride=}')

        n_samples = X.shape[0]
        if X.ndim == 4:
            # includes n_channels
            n_channels = X.shape[3]
        else:
            n_channels = 1

        image_size = X.shape[1]
        n_patches = (image_size - patch_size) // stride + 1

        nb = X.itemsize  # number of bytes each value

        new_shape = [n_samples,
                     n_patches,  # number of rows of patches
                     n_patches,  # number of columns of patches
                     patch_size,  # number of rows of pixels in each patch
                     patch_size,  # number of columns of pixels in each patch
                     n_channels]

        new_strides = [image_size * image_size * n_channels * nb,  # nuber of bytes to next image (sample)
                       image_size * stride * n_channels * nb,      # number of bytes to start of next patch in next row
                       stride * n_channels * nb,                   # number of bytes to start of next patch in next column
                       image_size * n_channels * nb,               # number of bytes to pixel in next row of patch
                       n_channels * nb,                            # number of bytes to pixel in next column of patch
                       nb]

        X = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

        # print(f'make_patches: Returning {X.shape=}')

        return X

    def _convolve(self, X, layer):
        """
        Convolves X and W

        Parameters
        ----------
        X : N x D x D x U
            N is number of samples
            D is number of rows and columns of input sample
            U is number of channels
        W: I x U
            I is 1 + number of weights in kernel
            U is number of units

        Returns
        -------
        NeuralNetwork object
        """

        debug = False

        if debug: print(f'convolve: {X.shape=} {layer["W"].shape=}') #  {kernel=} {stride=}')

        Xp = self._make_patches(X, layer['kernel'], layer['stride'])
        N, D, D, K, K, U = Xp.shape
        Xp = Xp.reshape(-1, K * K * U)

        W = layer['W']
        Uw = W.shape[1]

        XW = (Xp @ W[1:, :] + W[0:1, :])
        XW = XW.reshape(N, D, D, Uw)
        if debug: print(f'convolve: Returning {XW.shape=}\n')

        return XW  # , Xp.reshape(N, D, D, K, K, U)


    def _convolve_backprop(self, Delta, W, kernel, stride):
        """
        Back-propagate Delta through W in convolutional layer
        Pads Delta then convolves with W to back-propagate
        """

        debug = False
        
        N, D, _, U = Delta.shape
        n_zeros_edge = kernel - 1
        n_zeros_between = stride - 1
        # Start with zero array of correct size for DeltaZ
        DZrowcol = D + 2 * n_zeros_edge + n_zeros_between * D  # (D - 1)
        DeltaZ = np.zeros([N] + [DZrowcol] * 2 + [U])
        # copy Delta into correct positions of zero array

        N, Dz, Dz, U = DeltaZ.shape

        if debug: print(f'bp_cnn_convolve: {Delta.shape=} {W.shape=}, {kernel=}, {stride=}, {DeltaZ.shape=}')

        DeltaZ[:,
               n_zeros_edge:n_zeros_edge + Dz - 2 * n_zeros_edge:n_zeros_between + 1,
               n_zeros_edge:n_zeros_edge + Dz - 2 * n_zeros_edge:n_zeros_between + 1,
               :] = Delta
        DeltaZp = self._make_patches(DeltaZ, kernel, stride=1)  # use stride of 1, not actual layer stride value
        if debug: print(f'bp_cnn_convolve: {DeltaZp.shape=}')

        DZrowcolp = DeltaZp.shape[1]

        # ni, n_units = W.shape
        if debug: print(f'bp_cnn_convolve: {W.shape=}')
        n_units = W.shape[-1]
        W = W.reshape(kernel, kernel, -1, n_units)
        if debug: print(f'bp_cnn_convolve: reshaped {W.shape=}')
        W = np.swapaxes(W, 2, 3)
        if debug: print(f'bp_cnn_convolve: swapaxes {W.shape=}')

        W_flipped = np.flip(W, axis=(0, 1))

        if debug: print(f'bp_cnn_convolve: DeltaZp.reshaped {DeltaZp.reshape(N * DZrowcolp * DZrowcolp, kernel * kernel * n_units).shape} {W_flipped.reshape(kernel*kernel*n_units,-1).shape=}')

        Delta_bp = DeltaZp.reshape(N * DZrowcolp * DZrowcolp, kernel * kernel * n_units) @ W_flipped.reshape(kernel * kernel * n_units, -1)

        if debug: print(f'bp_cnn_convolve: {Delta_bp.shape=}')

        n = int(np.sqrt(Delta_bp.shape[0] / N))
        Delta_bp = Delta_bp.reshape(N, n, n, -1)

        if debug: print(f'bp_cnn_convolve: reshaped {Delta_bp.shape=}\n')

        return Delta_bp



    
if __name__ == '__main__':

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    T = np.array([[0], [1], [1], [0]])

    nnet1 = NeuralNetwork(2, [10], 1)
    nnet1.train(X, T, 100, 'adam', 0.01)
    print('\nRegression')
    print('      T                 Y')
    print(np.hstack((T, nnet1.use(X))))

    nnet2 = NeuralNetworkClassifier(2, [10], [0, 1])
    nnet2.train(X, T, 100, 'adam', 0.01)
    print('\nClassification')
    print('  T Y')
    print(np.hstack((T, nnet2.use(X)[0])))
    

    X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(4, 4, 4)
    T = np.array([[0], [1], [1], [0]])

    # nnet3 = NeuralNetworkClassifier_CNN([4, 4, 1], [(4, 2, 1), (2, 3, 1)], [5], 2)
    # nnet3 = NeuralNetworkClassifier_CNN([4, 4, 1], [(4, 2, 1), (2, 2, 1), (2, 2, 1)], [], 2)
    # nnet3 = NeuralNetworkClassifier_CNN([4, 4, 1], [(2, 3, 1)], [5], 2)
    # nnet3 = NeuralNetworkClassifier_CNN([4, 4, 1], [], [5, 5], 2)
    nnet3 = NeuralNetworkClassifier_CNN([4, 4, 1], [(10, 4, 1)], [50, 10], [0, 1])
    # X = X.reshape(4, -1)
    # nnet3 = NeuralNetworkClassifier(4*4, [10], [0, 1])
    nnet3.train(X, T, 500, 'adam', 0.01, verbose=True)
    print('\nCNN Classification')
    print('  T Y')
    print(np.hstack((T, nnet3.use(X)[0])))

