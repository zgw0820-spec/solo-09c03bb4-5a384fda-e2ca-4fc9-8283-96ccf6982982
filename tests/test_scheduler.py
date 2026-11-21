"""调度器单元测试：回填、预约、跨时刻推进、陈旧回报。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.scheduler import Scheduler, SchedulerError


def make_scheduler(racks: tuple[int, ...] = (4, 4)) -> Scheduler:
    return Scheduler(Settings(rack_capacities=racks))


def submit(s: Scheduler, nodes: int, duration: int) -> int:
    job, _ = s.submit(nodes, duration)
    return job.job_id


# --------------------------------------------------------------- 基础提交与占用


def test_submit_jobs_fill_racks_and_head_publishes_reservation():
    s = make_scheduler((4, 4))
    j1 = submit(s, 4, 10)
    j2 = submit(s, 4, 10)
    j3 = submit(s, 4, 5)

    j = s.jobs
    assert j[j1].status == "running" and j[j1].rack == "r1"
    assert j[j2].status == "running" and j[j2].rack == "r2"
    # 两个机架都满，队首等待并公布最早开工时刻
    assert j[j3].status == "queued"
    assert j[j3].scheduled_start == 10
    assert j[j3].rack in ("r1", "r2")
    assert len(j[j3].assigned_nodes) == 4
    occ = {r["rack"]: r for r in s.rack_occupancy()}
    assert occ["r1"]["used"] == occ["r2"]["used"] == 4
    assert {r["job_id"] for r in s.reservations()} == {j3}


def test_oversized_job_rejected():
    s = make_scheduler((3, 2))
    with pytest.raises(SchedulerError) as ei:
        s.submit(4, 1)
    assert ei.value.status_code == 422


# ----------------------------------------------------------------- 回填逻辑


def test_backfill_uses_free_space_without_delaying_reservation():
    s = make_scheduler((4, 4))
    submit(s, 4, 10)                  # 占满 r1
    submit(s, 3, 10)                  # r2 占用 0,1,2，节点 3 空闲
    waiting = submit(s, 4, 10)        # 队首：r1/r2 均在 t=10 才可凑齐
    assert s.jobs[waiting].scheduled_start == 10
    # 短作业：现在空位（r2 节点 3）上跑满时限也不晚于预约 → 回填
    short_ok = submit(s, 1, 3)
    # 跑满时限会超过预约时刻 → 不允许回填
    too_long = submit(s, 1, 12)

    j = s.jobs
    assert j[short_ok].status == "running"
    assert j[short_ok].rack == "r2"
    assert j[short_ok].assigned_nodes == [3]
    assert j[too_long].status == "queued"
    assert j[too_long].scheduled_start is None
    # 预约仍为 10，未被推迟
    assert j[waiting].scheduled_start == 10


def test_backfill_boundary_duration_equals_reservation_time_allowed():
    s = make_scheduler((4, 4))
    submit(s, 4, 10)                  # r1 满
    submit(s, 3, 10)                  # r2 节点 0,1,2
    waiting = submit(s, 4, 10)        # 预约 t=10
    # now=0，duration=10 => 完工时刻 10 <= 预约 10，边界放行
    exact = submit(s, 1, 10)
    assert s.jobs[exact].status == "running"
    assert s.jobs[exact].rack == "r2" and s.jobs[exact].assigned_nodes == [3]
    assert s.jobs[waiting].scheduled_start == 10


def test_backfill_cannot_take_reserved_nodes():
    s = make_scheduler((4,))
    submit(s, 3, 10)                  # r1 节点 0,1,2 被占
    head = submit(s, 4, 10)           # 需 4 节点，预约 t=10，含节点 3
    head_job = s.jobs[head]
    assert head_job.scheduled_start == 10
    assert set(head_job.assigned_nodes) == {0, 1, 2, 3}
    # 节点 3 当前空闲但已被预约，短作业不得占用
    thief = submit(s, 1, 5)
    assert s.jobs[thief].status == "queued"
    assert s.rack_occupancy()[0]["used"] == 3


# ---------------------------------------------------- 已公布时刻只可提前 + 换机架


def test_head_can_switch_rack_and_start_earlier_than_published():
    s = make_scheduler((3, 3))
    j1 = submit(s, 3, 10)             # r1 全占，t=10 释放
    submit(s, 2, 4)                   # r2 节点 0,1，t=4 释放
    head = submit(s, 3, 10)           # 队首：r2 t=4 可凑齐
    h = s.jobs[head]
    assert h.scheduled_start == 4 and h.rack == "r2"
    # r1 作业在 t=2 提前完工：r1 此刻即可凑齐 3 节点 → 立即开工、换到 r1
    s.advance(2)
    s.report_completion(j1, 2)
    assert h.status == "running"
    assert h.actual_start == 2
    assert h.rack == "r1"
    # 实际开工（2）早于此前公布的时刻（4），公布值本身从不推后
    assert h.actual_start <= h.scheduled_start == 4


def test_head_starts_immediately_when_space_frees():
    s = make_scheduler((4, 4))
    j1 = submit(s, 4, 10)
    submit(s, 4, 10)
    j3 = submit(s, 4, 10)
    assert s.jobs[j3].scheduled_start == 10
    # t=5 完工释放 r1，队首立即开工（早于预约 10）
    s.advance(5)
    s.report_completion(j1, 5)
    h = s.jobs[j3]
    assert h.status == "running"
    assert h.actual_start == 5 and h.rack == "r1"
    assert h.scheduled_start == 10      # 历史预约保留


# ------------------------------------------------------- 跨时刻推进与时限释放


def test_advance_crossing_multiple_release_points_reschedules_each():
    s = make_scheduler((4, 4))
    a = submit(s, 4, 2)                # r1，t=2 时限
    b = submit(s, 4, 4)                # r2，t=4 时限
    c = submit(s, 4, 10)               # 预约 r1 t=2
    d = submit(s, 1, 2)                # 暂无空位，排队
    assert s.jobs[c].scheduled_start == 2
    assert s.jobs[d].status == "queued"

    # 一次推进直接跨过 t=2、t=4、t=6 三个释放时刻
    _, target, events = s.advance(6)
    assert target == 6
    types = [e.type for e in events]
    # t=2 a 超时、c 开工；t=4 b 超时、d 开工；t=6 d 超时
    assert types.count("job_timeout") == 3
    assert types.count("job_started") == 2
    jc, jd = s.jobs[c], s.jobs[d]
    assert jc.status == "running" and jc.actual_start == 2   # 时限 t=12 未到
    assert jd.actual_start == 4
    assert jd.status == "killed" and jd.finish_time == 6
    assert s.rack_occupancy()[1]["used"] == 0


def test_timeout_kills_and_releases_then_reschedules():
    s = make_scheduler((4,))
    j = submit(s, 2, 3)
    s.advance(3)
    assert s.jobs[j].status == "killed"
    assert s.jobs[j].finish_reason == "killed"
    assert s.jobs[j].finish_time == 3
    assert s.rack_occupancy()[0]["used"] == 0


def test_backfill_started_during_advance_also_times_out_within_window():
    # 重排中新启动的作业，其时限也落在推进窗口内时应被继续处理
    s = make_scheduler((2, 2))
    submit(s, 2, 2)                    # r1，t=2 释放
    submit(s, 2, 8)                    # r2，t=8 释放
    submit(s, 2, 5)                    # 预约 r1 t=2，t=7 到期
    short = submit(s, 1, 2)            # 无空位，排队
    s.advance(6)
    # t=2 队首占 r1，r2 仍满 → short 继续等待
    assert s.jobs[short].status == "queued"
    s.advance(3)                       # 目标 t=9：t=7、t=8 各有释放
    assert s.jobs[short].actual_start == 7
    assert s.jobs[short].status == "killed"
    assert s.jobs[short].finish_time == 9


# ----------------------------------------------------------- 迟到完工回报处理


def test_late_completion_after_timeout_does_not_touch_reassigned_nodes():
    s = make_scheduler((4,))
    old = submit(s, 2, 3)
    s.advance(3)                        # 时限杀掉，节点释放
    assert s.jobs[old].status == "killed"
    new = submit(s, 2, 5)              # 节点转配给新作业
    assert s.jobs[new].status == "running"
    with pytest.raises(SchedulerError) as ei:
        s.report_completion(old, 3)    # 迟到回报被拒
    assert ei.value.status_code == 409
    # 新作业的节点占用原样保留
    assert s.jobs[new].status == "running"
    assert s.rack_occupancy()[0]["used"] == 2
    assert all(v == new for v in s.owners["r1"].values())


def test_duplicate_completion_is_stale():
    s = make_scheduler((4,))
    j = submit(s, 2, 5)
    s.advance(2)
    s.report_completion(j, 2)
    assert s.jobs[j].status == "finished"
    with pytest.raises(SchedulerError):
        s.report_completion(j, 2)


def test_completion_at_cannot_be_in_future_or_before_start():
    s = make_scheduler((4,))
    j = submit(s, 2, 5)
    with pytest.raises(SchedulerError):
        s.report_completion(j, 3)       # 当前时刻为 0，回报未来时刻
    s.advance(4)
    job, _ = s.report_completion(j, 4)  # 与时钟一致且未超时限，放行
    assert job.status == "finished"


def test_completion_beyond_deadline_rejected():
    s = make_scheduler((4,))
    j = submit(s, 2, 3)
    s.advance(3)                        # 已超时
    with pytest.raises(SchedulerError):
        s.report_completion(j, 3)


# ---------------------------------------------------------------- 杂项


def test_reset_clears_everything():
    s = make_scheduler((4,))
    submit(s, 4, 10)
    submit(s, 1, 2)
    s.advance(2)
    s.reset()
    assert s.now == 0
    assert s.jobs == {} and s.queue == []
    assert all(m == {} for m in s.owners.values())


def test_event_log_is_bounded():
    s = make_scheduler((4,))
    for _ in range(Scheduler.MAX_EVENTS + 50):
        s.reset()
        submit(s, 1, 1)
    assert len(s._events) <= Scheduler.MAX_EVENTS


# --------------------------------------------- 超时索引：只随运行作业数增长

def test_early_completions_keep_timeout_index_bounded_to_running_jobs():
    # 多个回填作业接连提前完工：它们原定的超时时刻不应留在索引里
    s = make_scheduler((4,))
    early_ids = [submit(s, 1, 10) for _ in range(2)]
    # 后两个作业时限更晚：本次推进的目标时刻之前不会到期
    long_ids = [submit(s, 1, 15) for _ in range(2)]
    assert len(s._timeout_heap) == 4          # 与运行作业数一致

    # 接连两次提前完工，每完成一个索引就同步收缩
    s.advance(2)
    s.report_completion(early_ids[0], 2)
    assert len(s._timeout_heap) == 3
    assert early_ids[0] not in s._timeout_heap._pos
    s.report_completion(early_ids[1], 2)
    assert len(s._timeout_heap) == 2
    assert early_ids[1] not in s._timeout_heap._pos
    # 完工记录仍可查询
    assert s.jobs[early_ids[0]].status == "finished"
    assert s.jobs[early_ids[0]].finish_time == 2

    # 跨过两个原定超时时刻 t=10（已随完工摘除，属陈旧时刻）到 t=12：
    # 剩余作业 t=15 才到期，窗口内没有任何节点释放；
    # 旧实现会在 t=10 因陈旧堆条目空转一次重排，修复后只在目标时刻重排一次
    calls = {"n": 0}
    original = s.reschedule

    def counting_reschedule():
        calls["n"] += 1
        return original()

    s.reschedule = counting_reschedule
    try:
        _, target, events = s.advance(10)
    finally:
        s.reschedule = original
    assert target == 12
    assert calls["n"] == 1
    assert [e.type for e in events].count("job_timeout") == 0
    # 索引仍只包含两个在跑的长作业，历史完工作业不占空间
    assert len(s._timeout_heap) == 2
    assert set(s._timeout_heap._pos) == set(long_ids)
    assert all(s.jobs[j].status == "running" for j in long_ids)
    # 提前完工者的历史记录仍可查询
    assert s.jobs[early_ids[0]].status == "finished"
    assert s.jobs[early_ids[1]].finish_reason == "completed"


def test_timeout_index_size_tracks_running_jobs_through_mixed_history():
    s = make_scheduler((4,))
    # 先制造一批已完工的历史作业
    old = [submit(s, 1, 2) for _ in range(4)]
    s.advance(1)
    s.report_completion(old[0], 1)
    s.report_completion(old[1], 1)
    s.advance(2)                                # 剩余两个时限被杀
    running = [j for j in s.jobs.values() if j.status == "running"]
    assert running == []
    # 历史作业再多，超时索引也已清空（不随作业历史增长）
    assert len(s._timeout_heap) == 0
    assert s._timeout_heap._pos == {}

    # 新一轮作业：索引规模始终等于当前运行作业数
    fresh = [submit(s, 1, 5) for _ in range(3)]
    running = [j for j in s.jobs.values() if j.status == "running"]
    assert len(s._timeout_heap) == len(running) == 3
    s.advance(3)
    s.report_completion(fresh[0], 3)
    running = [j for j in s.jobs.values() if j.status == "running"]
    assert len(s._timeout_heap) == len(running) == 2
    s.advance(5)                                # 余下两个在 t=8 时限到期
    assert len(s._timeout_heap) == 0


def test_same_time_timeouts_release_all_in_order_then_reschedule_once():
    # 四个作业同一时刻到期：按开工顺序全部释放后只重排一次
    s = make_scheduler((4,))
    ids = [submit(s, 1, 5) for _ in range(4)]
    calls = {"n": 0}
    original = s.reschedule

    def counting_reschedule():
        calls["n"] += 1
        return original()

    s.reschedule = counting_reschedule
    try:
        _, target, events = s.advance(5)
    finally:
        s.reschedule = original
    assert target == 5
    timeouts = [e for e in events if e.type == "job_timeout"]
    assert [e.job_id for e in timeouts] == ids  # 同刻按原顺序全部释放
    assert calls["n"] == 1                       # 全部释放后只重排一次
    assert all(s.jobs[j].status == "killed"
               and s.jobs[j].finish_time == 5 for j in ids)
    assert len(s._timeout_heap) == 0


def test_job_started_during_advance_due_at_target_is_processed_too():
    # 推进途中开工的作业，时限恰好落在目标时刻，也要在本次推进中处理
    s = make_scheduler((4,))
    blocker = submit(s, 4, 2)                   # t=2 释放机架
    short = submit(s, 1, 3)                     # t=2 才开工 -> t=5 到期
    assert s.jobs[short].status == "queued"
    _, target, events = s.advance(5)
    assert target == 5
    types = [e.type for e in events]
    assert types.count("job_timeout") == 2      # blocker@2、short@5
    assert s.jobs[blocker].finish_time == 2
    assert s.jobs[short].actual_start == 2
    assert s.jobs[short].status == "killed"
    assert s.jobs[short].finish_time == 5
    assert len(s._timeout_heap) == 0


def test_timeout_heap_discard_preserves_heap_invariant_and_ordering():
    from app.scheduler import _TimeoutHeap

    h = _TimeoutHeap()
    # 同刻条目按序号区分；不同时刻按释放时刻排序
    h.add(10, 1, 101)
    h.add(10, 2, 102)
    h.add(5, 3, 103)
    h.add(8, 4, 104)
    assert len(h) == 4

    # 摘除中间记录后堆结构与位置表保持一致
    h.discard(104)
    assert 104 not in h._pos
    assert len(h) == 3
    for idx, entry in enumerate(h._heap):
        assert h._pos[entry[2]] == idx

    # 最早时刻 t=5 先弹出，随后 t=10 的两条按原序号顺序成批弹出
    assert h.earliest_time() == 5
    assert [e[2] for e in h.pop_due(5)] == [103]
    assert h.earliest_time() == 10
    assert [e[2] for e in h.pop_due(10)] == [101, 102]
    assert len(h) == 0 and h._pos == {}

    # 摘除不存在的作业是幂等空操作
    h.discard(999)
    h.add(7, 5, 105)
    h.clear()
    assert len(h) == 0 and h._pos == {}
