

def build_graph_train(cell, output_wrapper, inp):
    """ Add training graph.

    :param cell: RNN cell to use
    :param output_wrapper: Linear output mapping from cell to output
    :param inp: Input series
    :return: List of outputs
    """
    outputs, state = cell(cell.get_zero_state_for_inp(inp), inp)
    outputs = output_wrapper(outputs)
    return outputs


def build_graph_single_step(cell, output_wrapper):
    """ Create graph for applying a single cell step

    :param cell: RNN cell to use
    :param output_wrapper: Linear output mapping from cell to output
    :return: Input placeholder, input state placeholder, output tensor, state tensor after cell computation
    """
    # Input placeholder
    pl_x = cell.get_input_placeholder(1, 'inp_single_step')
    # Placeholder for state
    pl_state = cell.get_state_placeholder('state_single_step')

    output, next_state = cell(pl_state, pl_x)
    output = output_wrapper(output)
    output = output[0]

    return pl_x,  pl_state, output, next_state


def build_graph_init(cell, output_wrapper, inp):
    """ Create graph which computes final RNN state for a given series and final output

    :param cell: RNN cell to use
    :param output_wrapper: Linear output mapping from cell to output
    :param inp: Input series
    :return: Output of last step and final state tensor
    """
    outputs, state = cell(cell.get_zero_state_for_inp(inp), inp)
    return output_wrapper(outputs), state
