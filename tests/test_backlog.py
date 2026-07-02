# Tests for the SAFe WSJF backlog — pure logic, no network.
import backlog as b


def test_wsjf_formula():
    item = {"value": 8, "time_criticality": 5, "risk": 2, "job_size": 5}
    assert b.wsjf(item) == 3.0  # (8+5+2)/5
    assert (
        b.wsjf({"value": 10, "time_criticality": 0, "risk": 0, "job_size": 0}) == 10.0
    )  # size 0→1


def test_ranked_orders_by_wsjf_desc_and_filters_status():
    items = [
        {
            "id": "A",
            "value": 5,
            "time_criticality": 5,
            "risk": 5,
            "job_size": 1,
            "status": "todo",
        },  # 15
        {
            "id": "B",
            "value": 8,
            "time_criticality": 2,
            "risk": 2,
            "job_size": 8,
            "status": "todo",
        },  # 1.5
        {
            "id": "C",
            "value": 9,
            "time_criticality": 9,
            "risk": 9,
            "job_size": 3,
            "status": "done",
        },  # excluded
    ]
    r = b.ranked(items)
    assert [i["id"] for i in r] == ["A", "B"]  # C (done) filtered out, A>B by WSJF


def test_next_pi_respects_capacity():
    items = [
        {
            "id": "A",
            "value": 5,
            "time_criticality": 5,
            "risk": 5,
            "job_size": 1,
            "status": "todo",
        },  # 15, sz1
        {
            "id": "B",
            "value": 8,
            "time_criticality": 5,
            "risk": 2,
            "job_size": 5,
            "status": "todo",
        },  # 3, sz5
        {
            "id": "C",
            "value": 8,
            "time_criticality": 2,
            "risk": 2,
            "job_size": 13,
            "status": "todo",
        },  # ~0.9, sz13
    ]
    plan = b.next_pi(items, capacity=6)
    assert {i["id"] for i in plan} == {"A", "B"}  # 1+5=6 fits; C (13) excluded
    assert sum(i["job_size"] for i in plan) <= 6


def test_seed_backlog_loads():
    items = b._load()
    assert any(i["status"] == "done" for i in items)  # velocity history present
    assert all("job_size" in i for i in items)
