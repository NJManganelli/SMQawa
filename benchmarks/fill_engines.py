import boost_histogram as bh
import hist
import numpy as np
from hist.chunked import ChunkedHist

import qawa.process.wztau2lnu_inclusive as wzmod

MULTICELL_BUFFER_BYTES = 32 * 2**20


def target_categories(histos, key):
    categories = [ax for ax in histos[key].axes if isinstance(ax, hist.axis.StrCategory)]
    if any(len(ax) for ax in categories):
        raise ValueError(f"histogram {key!r} already has categories; each target is filled once per pass")
    return categories


def slot_axis_names(categories):
    return [ax.name for ax in categories[1:]]


def concatenated_block(block):
    channel_list = [channel for channel, _ in block]
    row_sets = [rows for _, rows in block]
    dtype = np.result_type(np.int16, np.min_scalar_type(len(channel_list)))
    cidx = np.repeat(np.arange(len(channel_list), dtype=dtype), [len(r) for r in row_sets])
    rows = np.concatenate([np.empty(0, dtype=np.intp)] + row_sets)
    return channel_list, cidx, rows


def output_like(target, channel_list, labels):
    positions = [{} for ax in target.axes if isinstance(ax, hist.axis.StrCategory)][1:]
    for label in labels:
        for p, name in enumerate(label):
            positions[p].setdefault(name, len(positions[p]))
    slot_index = [np.array([positions[p][label[p]] for label in labels], dtype=np.intp) for p in range(len(positions))]
    names = [channel_list] + [list(p) for p in positions]
    axes = []
    dense = []
    for ax in target.axes:
        if isinstance(ax, hist.axis.StrCategory):
            new = hist.axis.StrCategory(names[len(axes) - len(dense)], name=ax.name, label=ax.label,
                                        growth=ax.traits.growth, overflow=ax.traits.overflow)
            new.__dict__.update(ax.__dict__)
            axes.append(new)
        else:
            axes.append(ax)
            dense.append(ax)
    output = hist.Hist(*axes, storage=hist.storage.Weight())
    output.__dict__.update(target.__dict__)
    return output, slot_index, dense


def group_cells(pass_state, name, n_rows, k):
    key = name + "_buffer"
    need = n_rows * 2 * k
    if key not in pass_state or pass_state[key].size < need:
        pass_state[key] = np.empty(max(need, MULTICELL_BUFFER_BYTES // 8))
    return pass_state[key][:need].reshape(n_rows, 2 * k)


def write_columns(weight, group, rows, out):
    for j, k in enumerate(group):
        out[:, j] = weight(k, rows)


def group_size_for(n_rows):
    return max(1, MULTICELL_BUFFER_BYTES // (max(n_rows, 1) * 16))


def multicell(histos, key, values, labels, weight, exclude, pass_state):
    target_categories(histos, key)
    block = wzmod.channel_block(exclude, pass_state)
    if not block:
        return
    channel_list, cidx, rows = concatenated_block(block)
    output, slot_index, dense = output_like(histos[key], channel_list, labels)
    assert "channel_index" not in [ax.name for ax in dense]
    view = output.view(flow=True)
    gathered = {ax.name: values[ax.name][rows] for ax in dense}
    n_rows = len(rows)
    n_slots = len(labels)
    group_size = group_size_for(n_rows)
    for start in range(0, n_slots, group_size):
        group = np.arange(start, min(start + group_size, n_slots))
        k = len(group)
        cells = group_cells(pass_state, "multicell", n_rows, k)
        write_columns(weight, group, rows, cells[:, :k])
        np.multiply(cells[:, :k], cells[:, :k], out=cells[:, k:])
        transient = hist.Hist(
            hist.axis.Integer(0, len(channel_list), name="channel_index", underflow=False, overflow=False),
            *dense,
            storage=hist.storage.MultiCell(2 * k),
        )
        transient.fill(channel_index=cidx, **gathered, weight=cells)
        cell_view = transient.view(flow=True)
        index = (slice(0, len(channel_list)),) + tuple(s[group] for s in slot_index)
        view.value[index] = np.swapaxes(cell_view[:k], 0, 1)
        view.variance[index] = np.swapaxes(cell_view[k:], 0, 1)
        del transient, cell_view
    histos[key] = output


def multicell_pr20(histos, key, values, labels, weight, exclude, pass_state):
    target_categories(histos, key)
    block = wzmod.channel_block(exclude, pass_state)
    if not block:
        return
    channel_list = [channel for channel, _ in block]
    output, slot_index, dense = output_like(histos[key], channel_list, labels)
    view = output.view(flow=True)
    gathered = [{ax.name: values[ax.name][rows] for ax in dense} for _, rows in block]
    n_slots = len(labels)
    group_size = group_size_for(max(len(rows) for _, rows in block))
    for start in range(0, n_slots, group_size):
        group = np.arange(start, min(start + group_size, n_slots))
        k = len(group)
        transient = hist.Hist(hist.axis.StrCategory(channel_list, name="channel_label", overflow=False), *dense,
                              storage=hist.storage.MultiCell(2 * k))
        for (channel, rows), gathered_c in zip(block, gathered):
            cells = group_cells(pass_state, "multicell_pr20", len(rows), k)
            write_columns(weight, group, rows, cells[:, :k])
            np.multiply(cells[:, :k], cells[:, :k], out=cells[:, k:])
            transient.fill(channel_label=channel, **gathered_c, weight=cells)
        cell_view = transient.view(flow=True)
        index = (slice(0, len(channel_list)),) + tuple(s[group] for s in slot_index)
        view.value[index] = np.swapaxes(cell_view[:k], 0, 1)
        view.variance[index] = np.swapaxes(cell_view[k:], 0, 1)
        del transient, cell_view
    histos[key] = output


def multicell_string(histos, key, values, labels, weight, exclude, pass_state):
    target_categories(histos, key)
    block = wzmod.channel_block(exclude, pass_state)
    if not block:
        return
    channel_list, cidx, rows = concatenated_block(block)
    output, slot_index, dense = output_like(histos[key], channel_list, labels)
    view = output.view(flow=True)
    channel_labels = np.array(channel_list, dtype=object)[cidx]
    gathered = {ax.name: values[ax.name][rows] for ax in dense}
    n_rows = len(rows)
    n_slots = len(labels)
    group_size = group_size_for(n_rows)
    for start in range(0, n_slots, group_size):
        group = np.arange(start, min(start + group_size, n_slots))
        k = len(group)
        cells = group_cells(pass_state, "multicell_string", n_rows, k)
        write_columns(weight, group, rows, cells[:, :k])
        np.multiply(cells[:, :k], cells[:, :k], out=cells[:, k:])
        transient = hist.Hist(hist.axis.StrCategory(channel_list, name="channel_label", overflow=False), *dense,
                              storage=hist.storage.MultiCell(2 * k))
        transient.fill(channel_label=channel_labels, **gathered, weight=cells)
        cell_view = transient.view(flow=True)
        index = (slice(0, len(channel_list)),) + tuple(s[group] for s in slot_index)
        view.value[index] = np.swapaxes(cell_view[:k], 0, 1)
        view.variance[index] = np.swapaxes(cell_view[k:], 0, 1)
        del transient, cell_view
    histos[key] = output


def chunked(histos, key, values, labels, weight, exclude, pass_state):
    target = histos[key]
    slot_names = slot_axis_names(target_categories(histos, key))
    dense_names = [ax.name for ax in target.axes if not isinstance(ax, hist.axis.StrCategory)]
    sink = ChunkedHist(*target.axes, storage=bh.storage.Weight())
    for channel, rows in wzmod.channel_block(exclude, pass_state):
        gathered = {name: values[name][rows] for name in dense_names}
        for k, label in enumerate(labels):
            sink.fill(channel=channel, **dict(zip(slot_names, label)), **gathered, weight=weight(k, rows))
    if not sink.chunks:
        return
    dense = sink.to_hist()
    axes = []
    for ax, dax in zip(target.axes, dense.axes):
        if isinstance(ax, hist.axis.StrCategory):
            new = hist.axis.StrCategory(list(dax), name=ax.name, label=ax.label,
                                        growth=ax.traits.growth, overflow=ax.traits.overflow)
            new.__dict__.update(ax.__dict__)
            axes.append(new)
        else:
            axes.append(ax)
    output = hist.Hist(*axes, storage=hist.storage.Weight())
    output.__dict__.update(target.__dict__)
    output.view(flow=True)[...] = dense.view(flow=True)
    histos[key] = output


def integer_axis(histos, key, values, labels, weight, exclude, pass_state):
    target_categories(histos, key)
    block = wzmod.channel_block(exclude, pass_state)
    if not block:
        return
    channel_list, cidx, rows = concatenated_block(block)
    output, slot_index, dense = output_like(histos[key], channel_list, labels)
    view = output.view(flow=True)
    gathered = {ax.name: values[ax.name][rows] for ax in dense}
    n_rows = len(rows)
    n_slots = len(labels)
    group_size = group_size_for(n_rows)
    for start in range(0, n_slots, group_size):
        group = np.arange(start, min(start + group_size, n_slots))
        k = len(group)
        cells = group_cells(pass_state, "integer_axis", n_rows, k)
        write_columns(weight, group, rows, cells[:, :k])
        transient = hist.Hist(
            hist.axis.Integer(0, len(channel_list), name="channel_index", underflow=False, overflow=False),
            hist.axis.Integer(0, k, name="slot_index", underflow=False, overflow=False),
            *dense, storage=hist.storage.Weight(),
        )
        transient.fill(
            channel_index=np.repeat(cidx, k),
            slot_index=np.tile(np.arange(k), n_rows),
            **{name: np.repeat(v, k) for name, v in gathered.items()},
            weight=np.ascontiguousarray(cells[:, :k]).reshape(-1),
        )
        tv = transient.view(flow=True)
        index = (slice(0, len(channel_list)),) + tuple(s[group] for s in slot_index)
        view.value[index] = tv["value"]
        view.variance[index] = tv["variance"]
        del transient, tv
    histos[key] = output


ENGINES = {
    "multicell": multicell,
    "multicell_pr20": multicell_pr20,
    "multicell_string": multicell_string,
    "chunked": chunked,
    "integer_axis": integer_axis,
}

ENGINE_DESCRIPTIONS = {
    "multicell": "phase-A MultiCell: Integer channel index axis, one fill per target per slot group",
    "multicell_pr20": "MultiCell, PR #20 layout: pre-declared StrCategory channel, one fill per (channel, slot group)",
    "multicell_string": "MultiCell, pre-declared StrCategory channel, one fill per slot group with a label array",
    "chunked": "ChunkedHist as a per-fill sink (scalar chunk keys only), converted per pass with to_hist()",
    "integer_axis": "reference row only, not a candidate (user decision D2): transient Integer x Integer Weight hist",
}
