"""Contracts every RL training loop here must satisfy (paper Methods p.7, Supplementary p.4, 3c).

Each contract runs against every implementation in IMPLEMENTATIONS. The frozen A3C loop
(`grid_cells/rl_a3c.py`) violates several of them on purpose — it is kept unchanged as the
baseline — so those cases are `xfail(strict=True)`: they must keep failing, and the day one
starts passing, the strict xfail turns red and says so.
"""
import pytest
import torch

import rl_harness

A3C_DEFECTS = {
    "reward_feedback": "B: reward input is never updated (always 0)",
    "bootstrap_next_state": "A: bootstrap uses V(s_t) instead of V(s_t+n)",
    "episode_reset": "C/D: goal code kept across episodes, grid state re-initialized from vision",
    "goal_is_not_terminal": "E: goal arrival resets the policy LSTM and cuts the return",
    "grid_state_continues_through_teleport": "C: grid state re-initialized at teleports",
    "sequences_span_teleports": "G: replayed grid sequences are cut at teleports",
}

IMPLEMENTATIONS = {"a3c": rl_harness.run_a3c}
if hasattr(rl_harness, "run_a2c"):
    IMPLEMENTATIONS["a2c"] = rl_harness.run_a2c

_traces = {}


def trace_for(impl):
    if impl not in _traces:
        _traces[impl] = IMPLEMENTATIONS[impl]()
    return _traces[impl]


def contract(name):
    def decorate(fn):
        params = []
        for impl in IMPLEMENTATIONS:
            marks = []
            if impl == "a3c" and name in A3C_DEFECTS:
                marks.append(pytest.mark.xfail(strict=True, reason=A3C_DEFECTS[name]))
            params.append(pytest.param(impl, marks=marks, id=impl))
        return pytest.mark.parametrize("impl", params)(fn)
    return decorate


def calls_by_env(trace):
    out = {}
    for call in trace.calls:
        out.setdefault(call.env, []).append(call)
    return out


@contract("reward_feedback")
def test_reward_feedback(impl):
    trace = trace_for(impl)
    checked = 0
    for calls in calls_by_env(trace).values():
        for prev, cur in zip(calls, calls[1:]):
            info = trace.registry[cur.obs_id]
            if info.first_of_episode:
                continue
            expected = rl_harness.GOAL_REWARD if info.after_goal else 0.0
            assert cur.reward_in == expected, (cur.obs_id, cur.reward_in, expected)
            checked += info.after_goal
    assert checked >= 2


@contract("bootstrap_next_state")
def test_bootstrap_next_state(impl):
    trace = trace_for(impl)
    checked = 0
    for w in trace.windows:
        if w.cut[-1]:
            continue
        assert w.bootstrap == w.value_ids[-1] + trace_next_offset(trace, w), \
            (w.value_ids, w.bootstrap)
        checked += 1
    assert checked >= 2


def trace_next_offset(trace, window):
    later = sorted(i for i, info in trace.registry.items()
                   if i > window.value_ids[-1] and env_of(trace, i) == window.env)
    return later[0] - window.value_ids[-1]


def env_of(trace, obs_id):
    for call in trace.calls:
        if call.obs_id == obs_id:
            return call.env
    return None


@contract("episode_reset")
def test_episode_reset(impl):
    trace = trace_for(impl)
    starts = 0
    for calls in calls_by_env(trace).values():
        for call in calls[1:]:
            if not trace.registry[call.obs_id].first_of_episode:
                continue
            assert torch.count_nonzero(call.goal_in) == 0, "goal grid code not cleared"
            assert call.prev_action_in == 0 and call.reward_in == 0.0
            assert torch.count_nonzero(call.ac_h_in) == 0, "policy LSTM not reset"
            starts += 1
    for g in trace.grid_calls:
        if g.obs_id is not None and trace.registry[g.obs_id].first_of_episode:
            assert torch.count_nonzero(g.h_in) == 0, "grid LSTM must start each episode at zero"
    assert starts >= 1


@contract("goal_is_not_terminal")
def test_goal_is_not_terminal(impl):
    trace = trace_for(impl)
    checked = 0
    for calls in calls_by_env(trace).values():
        for prev, cur in zip(calls, calls[1:]):
            if trace.registry[cur.obs_id].after_goal:
                assert torch.equal(cur.ac_h_in, prev.ac_h_out), "policy LSTM reset at a goal"
                checked += 1
    for w in trace.windows:
        for oid, cut in zip(w.value_ids, w.cut):
            if cut:
                nxt = [i for i, info in trace.registry.items() if i > oid]
                assert not nxt or trace.registry[min(nxt)].first_of_episode or \
                    not trace.registry[min(nxt)].after_goal, "return cut at a goal arrival"
    assert checked >= 2


@contract("grid_state_continues_through_teleport")
def test_grid_state_continues_through_teleport(impl):
    trace = trace_for(impl)
    by_env = {}
    for g in trace.grid_calls:
        by_env.setdefault(g.env, []).append(g)
    checked = 0
    for calls in by_env.values():
        for prev, cur in zip(calls, calls[1:]):
            if cur.obs_id is not None and trace.registry[cur.obs_id].after_goal:
                assert torch.equal(cur.h_in, prev.h_out), "grid LSTM re-initialized at teleport"
                checked += 1
    assert checked >= 2


@contract("sequences_span_teleports")
def test_sequences_span_teleports(impl):
    trace = trace_for(impl)
    assert trace.sequences
    spans_goal = False
    for seq in trace.sequences:
        episodes = {trace.registry[i].episode for i in seq}
        assert len(episodes) == 1, "a replayed sequence crosses an episode boundary"
        spans_goal |= any(trace.registry[i].after_goal for i in seq[1:])
    assert spans_goal, "no replayed grid sequence contains a teleport"
