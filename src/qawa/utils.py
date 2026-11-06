import copy

import hist
import numpy as np

from qawa import lumi_record

KIND_KEY = "kind"


def n_minus_one_key(requirements, exclude):
    sel_args = {s.replace('~', ''): '~' not in s for s in requirements if all(e not in s for e in exclude)}
    return tuple(sorted(sel_args.items()))


def n_minus_one_collisions(variables, selection_names):
    return sorted({(v, s) for v in variables for s in selection_names if v != s and v in s})


class ChannelSelections:
    def __init__(self, selection, channels):
        self.selection = selection
        self.channels = channels
        self.cache = {}

    def selected_event_indices(self, n_minus_one_vars):
        key = tuple((name, n_minus_one_key(reqs, n_minus_one_vars)) for name, reqs in self.channels.items())
        selected = self.cache.get(key)
        if selected is None:
            selected = tuple(
                (name, selected_event_indices) for name, selected_event_indices in
                ((name, np.flatnonzero(self.selection.require(**dict(channel_key)))) for name, channel_key in key)
                if len(selected_event_indices)
            )
            self.cache[key] = selected
        return selected


def syst_weight(weight_syst_stack, scale=None):
    if scale is None:
        def weight(syst_index, selected_event_indices):
            return np.nan_to_num(weight_syst_stack[syst_index][selected_event_indices], nan=1.0, posinf=1.0, neginf=1.0)
    else:
        def weight(syst_index, selected_event_indices):
            return np.nan_to_num(weight_syst_stack[syst_index][selected_event_indices] * scale[selected_event_indices],
                                 nan=1.0, posinf=1.0, neginf=1.0)
    return weight


def syst_weight_eft_struct_const(weight_syst_stack, eft_struct_consts):
    def weight(slot, selected_event_indices):
        syst_index, struct_const_index = divmod(slot, len(eft_struct_consts))
        return (np.nan_to_num(weight_syst_stack[syst_index][selected_event_indices], nan=1.0, posinf=1.0, neginf=1.0)
                * eft_struct_consts[struct_const_index][selected_event_indices])
    return weight


def fill_histogram(histograms_dict, hist_name, axes_variables, category_labels, syst_weight_fn, channel_selections):
    target = histograms_dict[hist_name]
    slot_names = [ax.name for ax in target.axes[1:] if isinstance(ax, hist.axis.StrCategory)]
    for channel, selected_event_indices in channel_selections.selected_event_indices(tuple(axes_variables)):
        dense = {name: v[selected_event_indices] for name, v in axes_variables.items()}
        for syst_index, label in enumerate(category_labels):
            target.fill(channel=channel, **dict(zip(slot_names, label)), **dense,
                        weight=syst_weight_fn(syst_index, selected_event_indices))


def merge_histograms(hists):
    first = hists[0]
    axes = []
    for i, ax in enumerate(first.axes):
        if isinstance(ax, hist.axis.StrCategory):
            labels = list(dict.fromkeys(label for h in hists for label in h.axes[i]))
            merged_ax = hist.axis.StrCategory(labels, growth=ax.traits.growth, overflow=ax.traits.overflow)
            merged_ax.__dict__.update(copy.deepcopy(ax.__dict__))
            axes.append(merged_ax)
        else:
            assert all(h.axes[i] == ax for h in hists[1:])
            axes.append(ax)
    assert all(h.storage_type == first.storage_type for h in hists[1:])
    out = type(first)(*axes, storage=first.storage_type())
    out.__dict__.update(copy.deepcopy(first.__dict__))
    out._variance_known = len(hists) == 1 and first._variance_known
    out_view = out.view(flow=True)
    for h in hists:
        if any(isinstance(ax, hist.axis.StrCategory) and len(ax) == 0 for ax in h.axes):
            continue
        index = []
        for ax, out_ax in zip(h.axes, out.axes):
            if isinstance(ax, hist.axis.StrCategory):
                index.append([out_ax.index(label) for label in ax] + ([len(out_ax)] if ax.traits.overflow else []))
            else:
                index.append(range(ax.extent))
        block = np.ix_(*index)
        view = h.view(flow=True)
        for field in view.dtype.names or (None,):
            if field is None:
                out_view[block] += view
            else:
                out_view[field][block] += view[field]
    return out


def merge_histogram_sets(outputs):
    grouped = {}
    for output in outputs:
        for dataset, histos in output.items():
            keys = grouped.setdefault(dataset, {})
            for key, h in histos.items():
                keys.setdefault(key, []).append(h)
    return {dataset: {key: merge_histograms(hists) for key, hists in keys.items()} for dataset, keys in grouped.items()}


def dataset_record(dataset, content, sumw):
    record = {"hist": content, "sumw": sumw}
    pairs = content.pop(lumi_record.LUMI_KEY, None)
    if pairs is not None:
        record[lumi_record.LUMI_KEY] = lumi_record.as_array(pairs)
    kinds = content.pop(KIND_KEY, None)
    if kinds is not None:
        if len(kinds) != 1:
            raise RuntimeError(f"dataset {dataset}: its chunks report several kinds {sorted(kinds)}")
        (record[KIND_KEY],) = kinds
    return record
