"""
Antigen processing neural network implementation - PyTorch version
"""

from __future__ import print_function

import time
import collections
import gc
import json
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .flanking_encoding import FlankingEncoding
from .common import configure_pytorch, get_pytorch_device


class Class1ProcessingModel(nn.Module):
    """
    PyTorch module for antigen processing prediction.
    """

    def __init__(
            self,
            sequence_dims,
            n_flank_length,
            c_flank_length,
            peptide_max_length,
            flanking_averages,
            convolutional_filters,
            convolutional_kernel_size,
            convolutional_activation,
            convolutional_kernel_l1_l2,
            dropout_rate,
            post_convolutional_dense_layer_sizes):
        super(Class1ProcessingModel, self).__init__()

        self.n_flank_length = n_flank_length
        self.c_flank_length = c_flank_length
        self.peptide_max_length = peptide_max_length
        self.flanking_averages = flanking_averages

        # Input channels from sequence encoding
        in_channels = sequence_dims[1]

        # Main convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=convolutional_filters,
            kernel_size=convolutional_kernel_size,
            padding='same'
        )

        # Activation function
        if convolutional_activation == 'tanh':
            self.conv_activation = torch.tanh
        elif convolutional_activation == 'relu':
            self.conv_activation = F.relu
        elif convolutional_activation == 'sigmoid':
            self.conv_activation = torch.sigmoid
        else:
            self.conv_activation = torch.tanh

        # Dropout
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        # Post-convolutional dense layers for each flank
        # These are implemented as 1D convolutions with kernel_size=1
        layer_sizes = list(post_convolutional_dense_layer_sizes) + [1]

        self.n_flank_post_convs = nn.ModuleList()
        self.c_flank_post_convs = nn.ModuleList()

        current_channels = convolutional_filters
        for i, size in enumerate(layer_sizes):
            self.n_flank_post_convs.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=size,
                kernel_size=1
            ))
            self.c_flank_post_convs.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=size,
                kernel_size=1
            ))
            current_channels = size

        # Dense layers for flanking averages (if enabled)
        self.n_flank_avg_dense = None
        self.c_flank_avg_dense = None
        if flanking_averages:
            if n_flank_length > 0:
                self.n_flank_avg_dense = nn.Linear(convolutional_filters, 1)
            if c_flank_length > 0:
                self.c_flank_avg_dense = nn.Linear(convolutional_filters, 1)

        # Final output layer
        # Number of inputs: 2 from n_flank (cleaved + max_pool) + 2 from c_flank
        # Plus optional flanking averages
        num_final_inputs = 4
        if flanking_averages and n_flank_length > 0:
            num_final_inputs += 1
        if flanking_averages and c_flank_length > 0:
            num_final_inputs += 1

        self.output_layer = nn.Linear(num_final_inputs, 1)
        # Initialize output weights to ones (like Keras initializers.Ones())
        nn.init.ones_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs):
        """
        Forward pass.

        Parameters
        ----------
        inputs : dict
            Dictionary with 'sequence' and 'peptide_length' keys

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch,)
        """
        sequence = inputs['sequence']  # (batch, seq_len, channels)
        peptide_length = inputs['peptide_length']  # (batch, 1)

        # Transpose for Conv1d: (batch, channels, seq_len)
        x = sequence.permute(0, 2, 1)

        # Apply main convolution
        x = self.conv1(x)
        x = self.conv_activation(x)

        if self.dropout is not None:
            # Spatial dropout: same dropout mask for all positions
            # Equivalent to Keras Dropout with noise_shape=(None, 1, channels)
            x = self.dropout(x)

        # Transpose back: (batch, seq_len, channels)
        convolutional_result = x.permute(0, 2, 1)

        outputs_for_final = []

        # Process n_flank
        n_flank_outputs = self._process_n_flank(
            convolutional_result, peptide_length
        )
        outputs_for_final.extend(n_flank_outputs)

        # Process c_flank
        c_flank_outputs = self._process_c_flank(
            convolutional_result, peptide_length
        )
        outputs_for_final.extend(c_flank_outputs)

        # Concatenate all outputs
        combined = torch.cat(outputs_for_final, dim=-1)

        # Final output
        output = torch.sigmoid(self.output_layer(combined))
        return output.squeeze(-1)

    def _process_n_flank(self, conv_result, peptide_length):
        """Process n_flank feature extraction."""
        outputs = []

        # Apply post-convolutional layers
        # Transpose for Conv1d
        x = conv_result.permute(0, 2, 1)
        for i, conv_layer in enumerate(self.n_flank_post_convs):
            x = conv_layer(x)
            if i < len(self.n_flank_post_convs) - 1:
                x = self.conv_activation(x)
            else:
                x = torch.tanh(x)  # Final layer always tanh
        # Transpose back
        single_output_result = x.permute(0, 2, 1)  # (batch, seq_len, 1)

        # Extract at cleavage position (n_flank_length)
        cleaved = single_output_result[:, self.n_flank_length, :]  # (batch, 1)
        outputs.append(cleaved)

        # Max pool over peptide (excluding first position)
        max_pool = self._max_pool_over_peptide_n(
            single_output_result, peptide_length
        )
        outputs.append(max_pool)

        # Optional flanking average
        if self.n_flank_avg_dense is not None and self.n_flank_length > 0:
            # Average over n_flank region
            n_flank_region = conv_result[:, :self.n_flank_length, :]  # (batch, n_flank_length, channels)
            avg = n_flank_region.mean(dim=1)  # (batch, channels)
            dense_out = torch.tanh(self.n_flank_avg_dense(avg))  # (batch, 1)
            outputs.append(dense_out)

        return outputs

    def _process_c_flank(self, conv_result, peptide_length):
        """Process c_flank feature extraction."""
        outputs = []

        # Apply post-convolutional layers
        x = conv_result.permute(0, 2, 1)
        for i, conv_layer in enumerate(self.c_flank_post_convs):
            x = conv_layer(x)
            if i < len(self.c_flank_post_convs) - 1:
                x = self.conv_activation(x)
            else:
                x = torch.tanh(x)
        single_output_result = x.permute(0, 2, 1)  # (batch, seq_len, 1)

        # Extract at cleavage position (dynamic based on peptide_length)
        cleaved = self._extract_c_cleavage(single_output_result, peptide_length)
        outputs.append(cleaved)

        # Max pool over peptide (excluding last position)
        max_pool = self._max_pool_over_peptide_c(
            single_output_result, peptide_length
        )
        outputs.append(max_pool)

        # Optional flanking average
        if self.c_flank_avg_dense is not None and self.c_flank_length > 0:
            # Average over c_flank region (dynamic based on peptide_length)
            avg = self._extract_c_flank_avg(conv_result, peptide_length)
            dense_out = torch.tanh(self.c_flank_avg_dense(avg))
            outputs.append(dense_out)

        return outputs

    def _max_pool_over_peptide_n(self, x, peptide_length):
        """
        Max pool over peptide region excluding first position.
        For n_flank cleavage site.
        """
        batch_size, seq_len, features = x.shape
        peptide_length = peptide_length.view(-1)

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Mask: 1 for positions from n_flank_length+1 to n_flank_length+peptide_length
        starts = self.n_flank_length + 1
        ends = (self.n_flank_length + peptide_length).unsqueeze(1)
        mask = (positions >= starts) & (positions < ends)  # (batch, seq_len)

        # Apply mask (assuming x >= -1 from tanh)
        x_shifted = x + 1
        mask_expanded = mask.unsqueeze(-1).float()
        masked_x = x_shifted * mask_expanded
        max_value = masked_x.max(dim=1)[0] - 1  # (batch, features)

        # Flip sign
        return -1 * max_value

    def _max_pool_over_peptide_c(self, x, peptide_length):
        """
        Max pool over peptide region excluding last position.
        For c_flank cleavage site.
        """
        batch_size, seq_len, features = x.shape
        peptide_length = peptide_length.view(-1)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Mask: 1 for positions from n_flank_length to n_flank_length+peptide_length-1
        starts = self.n_flank_length
        ends = (self.n_flank_length + peptide_length - 1).unsqueeze(1)
        mask = (positions >= starts) & (positions < ends)

        x_shifted = x + 1
        mask_expanded = mask.unsqueeze(-1).float()
        masked_x = x_shifted * mask_expanded
        max_value = masked_x.max(dim=1)[0] - 1

        return -1 * max_value

    def _extract_c_cleavage(self, x, peptide_length):
        """Extract at c-terminal cleavage position."""
        peptide_length = peptide_length.view(-1)
        indices = self.n_flank_length + peptide_length - 1

        batch_size = x.size(0)
        indices = indices.long().view(batch_size, 1, 1).expand(-1, -1, x.size(2))
        result = x.gather(1, indices).squeeze(1)  # (batch, features)
        return result

    def _extract_c_flank_avg(self, conv_result, peptide_length):
        """Average over c_flank region."""
        batch_size, seq_len, features = conv_result.shape
        peptide_length = peptide_length.view(-1)

        positions = torch.arange(seq_len, device=conv_result.device).unsqueeze(0)

        # Mask: 1 for c_flank positions
        starts = (self.n_flank_length + peptide_length).unsqueeze(1)
        ends = starts + self.c_flank_length
        mask = (positions >= starts) & (positions < ends)

        x_shifted = conv_result + 1
        mask_expanded = mask.unsqueeze(-1).float()
        masked_x = x_shifted * mask_expanded
        sum_value = masked_x.sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        avg_value = sum_value / count - 1

        return avg_value

    def get_weights_list(self):
        """Get weights as a list of numpy arrays."""
        weights = []
        for name, param in self.named_parameters():
            weights.append(param.detach().cpu().numpy())
        for name, buffer in self.named_buffers():
            weights.append(buffer.detach().cpu().numpy())
        return weights

    def set_weights_list(self, weights, auto_convert_keras=True):
        """
        Set weights from a list of numpy arrays.

        Supports automatic detection and conversion of Keras-format weights.

        Parameters
        ----------
        weights : list of numpy.ndarray
        auto_convert_keras : bool
            If True, automatically detect and convert Keras-format weights
        """
        # Keras stores weights in layer definition order which interleaves
        # n_flank and c_flank post-conv layers:
        #   conv1, n_post_0, c_post_0, n_post_1, c_post_1, n_avg, c_avg, output
        # PyTorch ModuleList stores:
        #   conv1, n_post_0, n_post_1, c_post_0, c_post_1, n_avg, c_avg, output
        # We need to reorder Keras weights to match PyTorch parameter order

        if auto_convert_keras:
            weights = self._reorder_keras_weights(list(weights))

        idx = 0
        for name, param in self.named_parameters():
            w = weights[idx].astype(numpy.float32)

            # Auto-detect and convert Keras weights if shapes don't match
            if auto_convert_keras and w.shape != param.shape:
                # Dense/Linear: Keras (in, out) -> PyTorch (out, in)
                if len(w.shape) == 2 and w.shape == param.shape[::-1]:
                    w = w.T
                # Conv1D: Keras (k, in_ch, out_ch) -> PyTorch (out_ch, in_ch, k)
                elif len(w.shape) == 3 and w.shape == (param.shape[2], param.shape[1], param.shape[0]):
                    w = w.transpose(2, 1, 0)

            if w.shape != param.shape:
                raise ValueError(
                    f"Weight shape mismatch for {name}: "
                    f"got {weights[idx].shape}, expected {param.shape}"
                )

            param.data = torch.from_numpy(w)
            idx += 1
        for name, buffer in self.named_buffers():
            self._buffers[name] = torch.from_numpy(weights[idx].astype(numpy.float32))
            idx += 1

    def _reorder_keras_weights(self, weights):
        """
        Reorder Keras weights to match PyTorch parameter order.

        Keras interleaves n_flank and c_flank post-conv layers:
            conv1, n_post_0, c_post_0, n_post_1, c_post_1, ..., n_avg, c_avg, output
        PyTorch has:
            conv1, n_post_0, n_post_1, ..., c_post_0, c_post_1, ..., n_avg, c_avg, output

        Returns
        -------
        list of numpy.ndarray
        """
        # Count how many post-conv layers there are (each has weight + bias)
        n_post_conv_layers = len(self.n_flank_post_convs)
        if n_post_conv_layers == 0:
            return weights

        # Find indices in Keras weight list
        # Structure: conv1_w, conv1_b, [n_post_i_w, n_post_i_b, c_post_i_w, c_post_i_b]...,
        #            n_avg_w, n_avg_b, c_avg_w, c_avg_b, out_w, out_b

        reordered = []

        # Conv1 weights (indices 0, 1)
        reordered.append(weights[0])
        reordered.append(weights[1])

        # Keras has interleaved: n_post_0, c_post_0, n_post_1, c_post_1, ...
        # We need: n_post_0, n_post_1, ..., c_post_0, c_post_1, ...
        post_conv_start = 2
        post_conv_end = post_conv_start + n_post_conv_layers * 4  # 4 = n_w, n_b, c_w, c_b per layer

        # Extract n_flank and c_flank post-conv weights separately
        n_flank_weights = []
        c_flank_weights = []
        for i in range(n_post_conv_layers):
            keras_idx = post_conv_start + i * 4
            n_flank_weights.append(weights[keras_idx])      # n_post_i_w
            n_flank_weights.append(weights[keras_idx + 1])  # n_post_i_b
            c_flank_weights.append(weights[keras_idx + 2])  # c_post_i_w
            c_flank_weights.append(weights[keras_idx + 3])  # c_post_i_b

        # Add in PyTorch order: all n_flank first, then all c_flank
        reordered.extend(n_flank_weights)
        reordered.extend(c_flank_weights)

        # Remaining weights (avg dense and output) stay in same order
        reordered.extend(weights[post_conv_end:])

        return reordered


class Class1ProcessingNeuralNetwork(object):
    """
    A neural network for antigen processing prediction
    """

    network_hyperparameter_defaults = HyperparameterDefaults(
        amino_acid_encoding="BLOSUM62",
        peptide_max_length=15,
        n_flank_length=10,
        c_flank_length=10,
        flanking_averages=False,
        convolutional_filters=16,
        convolutional_kernel_size=8,
        convolutional_activation="tanh",
        convolutional_kernel_l1_l2=[0.0001, 0.0001],
        dropout_rate=0.5,
        post_convolutional_dense_layer_sizes=[],
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=256,
    )
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=30,
        min_delta=0.0,
    )
    """
    Hyperparameters for early stopping.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        optimizer="adam",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters.
    """

    auxiliary_input_hyperparameter_defaults = HyperparameterDefaults()
    """
    Allele feature hyperparameters.
    """

    hyperparameter_defaults = (
        network_hyperparameter_defaults.extend(fit_hyperparameter_defaults)
        .extend(early_stopping_hyperparameter_defaults)
        .extend(compile_hyperparameter_defaults)
        .extend(auxiliary_input_hyperparameter_defaults)
    )

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters
        )
        self._network = None
        self.network_json = None
        self.network_weights = None
        self.fit_info = []
        self._device = None

    @property
    def sequence_lengths(self):
        """
        Supported maximum sequence lengths

        Returns
        -------
        dict of string -> int

        Keys are "peptide", "n_flank", "c_flank". Values give the maximum
        supported sequence length.
        """
        return {
            "peptide": self.hyperparameters["peptide_max_length"],
            "n_flank": self.hyperparameters["n_flank_length"],
            "c_flank": self.hyperparameters["c_flank_length"],
        }

    def get_device(self):
        """Get the PyTorch device to use."""
        if self._device is None:
            self._device = get_pytorch_device()
        return self._device

    def network(self):
        """
        Return the PyTorch model associated with this network.
        """
        if self._network is None and self.network_json is not None:
            # Re-create the network using hyperparameters
            self._network = self.make_network(
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if self.network_weights is not None:
                # Detect if weights are from Keras or PyTorch format
                # Keras JSON has 'class_name': 'Model', PyTorch has 'hyperparameters'
                try:
                    config = json.loads(self.network_json)
                    is_keras_format = config.get('class_name') == 'Model'
                except (json.JSONDecodeError, TypeError):
                    is_keras_format = False
                self._network.set_weights_list(
                    self.network_weights,
                    auto_convert_keras=is_keras_format
                )
        return self._network

    def update_network_description(self):
        """
        Update self.network_json and self.network_weights properties based on
        this instances's neural network.
        """
        if self._network is not None:
            # Store hyperparameters as JSON (not the actual model structure)
            self.network_json = json.dumps({'hyperparameters': dict(self.hyperparameters)})
            self.network_weights = self._network.get_weights_list()

    def fit(
        self,
        sequences,
        targets,
        sample_weights=None,
        shuffle_permutation=None,
        verbose=1,
        progress_callback=None,
        progress_preamble="",
        progress_print_interval=5.0,
    ):
        """
        Fit the neural network.

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and upstream/downstream flanking sequences
        targets : list of float
            1 indicates hit, 0 indicates decoy
        sample_weights : list of float
            If not specified all samples have equal weight.
        shuffle_permutation : list of int
            Permutation (integer list) of same length as peptides and affinities
            If None, then a random permutation will be generated.
        verbose : int
            Verbosity level
        progress_callback : function
            No-argument function to call after each epoch.
        progress_preamble : string
            Optional string of information to include in each progress update
        progress_print_interval : float
            How often (in seconds) to print progress update. Set to None to
            disable.
        """
        configure_pytorch()
        device = self.get_device()

        x_dict = self.network_input(sequences)

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(targets))
        targets = numpy.array(targets)[shuffle_permutation]
        assert numpy.isnan(targets).sum() == 0, targets
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights)[shuffle_permutation]
        for key in list(x_dict):
            x_dict[key] = x_dict[key][shuffle_permutation]

        fit_info = collections.defaultdict(list)

        if self._network is None:
            self._network = self.make_network(
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > -1:
                print(self._network)

        network = self.network()
        network.to(device)

        # Setup optimizer
        optimizer = self._create_optimizer(network)

        # Loss function (binary cross-entropy)
        loss_fn = nn.BCELoss(reduction='none')

        # Validation split
        val_split = self.hyperparameters["validation_split"]
        n_total = len(targets)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        indices = numpy.arange(n_total)
        numpy.random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        last_progress_print = None
        min_val_loss_iteration = None
        min_val_loss = None
        start = time.time()

        for epoch in range(self.hyperparameters["max_epochs"]):
            epoch_start = time.time()
            network.train()

            # Shuffle training indices each epoch
            numpy.random.shuffle(train_indices)

            batch_size = self.hyperparameters["minibatch_size"]
            train_losses = []

            for batch_start in range(0, n_train, batch_size):
                batch_idx = train_indices[batch_start:batch_start + batch_size]

                seq_batch = torch.from_numpy(x_dict["sequence"][batch_idx]).float().to(device)
                length_batch = torch.from_numpy(x_dict["peptide_length"][batch_idx]).to(device)
                target_batch = torch.from_numpy(targets[batch_idx].astype(numpy.float32)).to(device)

                inputs = {"sequence": seq_batch, "peptide_length": length_batch}

                optimizer.zero_grad()
                predictions = network(inputs)
                loss = loss_fn(predictions, target_batch)

                if sample_weights is not None:
                    weight_batch = torch.from_numpy(
                        sample_weights[batch_idx].astype(numpy.float32)
                    ).to(device)
                    loss = loss * weight_batch

                loss = loss.mean()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            epoch_time = time.time() - epoch_start
            train_loss = numpy.mean(train_losses)
            fit_info["loss"].append(train_loss)

            # Validation
            if val_split > 0:
                network.eval()
                with torch.no_grad():
                    val_seq = torch.from_numpy(x_dict["sequence"][val_indices]).float().to(device)
                    val_length = torch.from_numpy(x_dict["peptide_length"][val_indices]).to(device)
                    val_targets = torch.from_numpy(targets[val_indices].astype(numpy.float32)).to(device)

                    val_inputs = {"sequence": val_seq, "peptide_length": val_length}
                    val_predictions = network(val_inputs)
                    val_loss = loss_fn(val_predictions, val_targets).mean().item()
                fit_info["val_loss"].append(val_loss)

            gc.collect()

            # Progress printing
            if progress_print_interval is not None and (
                not last_progress_print
                or (time.time() - last_progress_print > progress_print_interval)
            ):
                print(
                    (
                        progress_preamble
                        + " "
                        + "Epoch %3d / %3d [%0.2f sec]: loss=%g. "
                        "Min val loss (%s) at epoch %s"
                        % (
                            epoch,
                            self.hyperparameters["max_epochs"],
                            epoch_time,
                            fit_info["loss"][-1],
                            str(min_val_loss),
                            min_val_loss_iteration,
                        )
                    ).strip()
                )
                last_progress_print = time.time()

            if val_split > 0:
                if min_val_loss is None or (
                    val_loss < min_val_loss - self.hyperparameters["min_delta"]
                ):
                    min_val_loss = val_loss
                    min_val_loss_iteration = epoch

                if self.hyperparameters["early_stopping"]:
                    threshold = (
                        min_val_loss_iteration + self.hyperparameters["patience"]
                    )
                    if epoch > threshold:
                        if progress_print_interval is not None:
                            print(
                                (
                                    progress_preamble
                                    + " "
                                    + "Stopping at epoch %3d / %3d: loss=%g. "
                                    "Min val loss (%g) at epoch %s"
                                    % (
                                        epoch,
                                        self.hyperparameters["max_epochs"],
                                        fit_info["loss"][-1],
                                        (
                                            min_val_loss
                                            if min_val_loss is not None
                                            else numpy.nan
                                        ),
                                        min_val_loss_iteration,
                                    )
                                ).strip()
                            )
                        break

            if progress_callback:
                progress_callback()

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = len(sequences.dataframe)
        self.fit_info.append(dict(fit_info))

        if verbose > -1:
            print("Output weights", self.network().output_layer.weight.data.cpu().numpy())

    def _create_optimizer(self, network):
        """Create an optimizer for the network."""
        optimizer_name = self.hyperparameters["optimizer"].lower()
        lr = self.hyperparameters["learning_rate"] or 0.001

        # L1/L2 regularization is applied via weight_decay (L2 only in PyTorch)
        l1, l2 = self.hyperparameters.get("convolutional_kernel_l1_l2", [0.0, 0.0])

        if optimizer_name == "adam":
            return torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l2)
        elif optimizer_name == "rmsprop":
            return torch.optim.RMSprop(network.parameters(), lr=lr, weight_decay=l2)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(network.parameters(), lr=lr, weight_decay=l2)
        else:
            return torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l2)

    def predict(
        self,
        peptides,
        n_flanks=None,
        c_flanks=None,
        batch_size=DEFAULT_PREDICT_BATCH_SIZE,
    ):
        """
        Predict antigen processing.

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        n_flanks : list of string
            Upstream sequence before each peptide
        c_flanks : list of string
            Downstream sequence after each peptide
        batch_size : int
            Prediction batch size.

        Returns
        -------
        numpy.array

        Processing scores. Range is 0-1, higher indicates more favorable
        processing.
        """
        if n_flanks is None:
            n_flanks = [""] * len(peptides)
        if c_flanks is None:
            c_flanks = [""] * len(peptides)

        sequences = FlankingEncoding(
            peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks
        )
        return self.predict_encoded(sequences=sequences, batch_size=batch_size)

    def predict_encoded(
        self, sequences, throw=True, batch_size=DEFAULT_PREDICT_BATCH_SIZE
    ):
        """
        Predict antigen processing.

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and flanking sequences
        throw : boolean
            Whether to throw exception on unsupported peptides
        batch_size : int
            Prediction batch size.

        Returns
        -------
        numpy.array
        """
        configure_pytorch()
        device = self.get_device()

        x_dict = self.network_input(sequences, throw=throw)
        network = self.network()
        network.to(device)
        network.eval()

        n_samples = len(x_dict["sequence"])
        all_predictions = []

        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                seq_batch = torch.from_numpy(
                    x_dict["sequence"][batch_start:batch_end]
                ).float().to(device)
                length_batch = torch.from_numpy(
                    x_dict["peptide_length"][batch_start:batch_end]
                ).to(device)

                inputs = {"sequence": seq_batch, "peptide_length": length_batch}
                batch_predictions = network(inputs)
                all_predictions.append(batch_predictions.cpu().numpy())

        raw_predictions = numpy.concatenate(all_predictions, axis=0)
        predictions = numpy.array(raw_predictions, dtype="float64")
        return predictions

    def network_input(self, sequences, throw=True):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and flanking sequences
        throw : boolean
            Whether to throw exception on unsupported peptides

        Returns
        -------
        dict
        """
        encoded = sequences.vector_encode(
            self.hyperparameters["amino_acid_encoding"],
            self.hyperparameters["peptide_max_length"],
            n_flank_length=self.hyperparameters["n_flank_length"],
            c_flank_length=self.hyperparameters["c_flank_length"],
            allow_unsupported_amino_acids=True,
            throw=throw,
        )

        result = {
            "sequence": encoded.array,
            "peptide_length": encoded.peptide_lengths,
        }
        return result

    def make_network(
        self,
        amino_acid_encoding,
        peptide_max_length,
        n_flank_length,
        c_flank_length,
        flanking_averages,
        convolutional_filters,
        convolutional_kernel_size,
        convolutional_activation,
        convolutional_kernel_l1_l2,
        dropout_rate,
        post_convolutional_dense_layer_sizes,
    ):
        """
        Helper function to make a PyTorch network given hyperparameters.
        """
        configure_pytorch()

        empty_x_dict = self.network_input(FlankingEncoding([], [], []))
        sequence_dims = empty_x_dict["sequence"].shape[1:]

        numpy.testing.assert_equal(
            sequence_dims[0], peptide_max_length + n_flank_length + c_flank_length
        )

        return Class1ProcessingModel(
            sequence_dims=sequence_dims,
            n_flank_length=n_flank_length,
            c_flank_length=c_flank_length,
            peptide_max_length=peptide_max_length,
            flanking_averages=flanking_averages,
            convolutional_filters=convolutional_filters,
            convolutional_kernel_size=convolutional_kernel_size,
            convolutional_activation=convolutional_activation,
            convolutional_kernel_l1_l2=convolutional_kernel_l1_l2,
            dropout_rate=dropout_rate,
            post_convolutional_dense_layer_sizes=post_convolutional_dense_layer_sizes,
        )

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.

        Returns
        -------
        dict

        """
        self.update_network_description()
        result = dict(self.__dict__)
        result["_network"] = None
        result["_device"] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)
        self._device = None

    def get_weights(self):
        """
        Get the network weights

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        self.update_network_description()
        return self.network_weights

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        self.update_network_description()
        result = dict(self.__dict__)
        del result["_network"]
        result["network_weights"] = None
        result["_device"] = None
        return result

    @classmethod
    def from_config(cls, config, weights=None):
        """
        deserialize from a dict returned by get_config().

        Parameters
        ----------
        config : dict
        weights : list of array, optional
            Network weights to restore

        Returns
        -------
        Class1ProcessingNeuralNetwork
        """
        config = dict(config)
        instance = cls(**config.pop("hyperparameters"))
        instance.__dict__.update(config)
        instance.network_weights = weights
        assert instance._network is None
        return instance
