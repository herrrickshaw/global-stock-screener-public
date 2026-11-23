import dataclasses
import logging
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.market_data.domain.models import OHLC, MarketCalendar, MarketData, Metric, Price, Ticker
from src.shared.domain import events as ev
from src.shared.domain.base import (
    AggregateRoot,
    ApplicationService,
    DomainEvent,
    DomainException,
    Repository,
    ValueObject,
)
from src.shared.infrastructure import event_bus as eb


# ---------------- base ----------------
class Agg(AggregateRoot):
    pass


def test_domain_event_defaults():
    a, b = DomainEvent(), DomainEvent()
    assert a.event_id != b.event_id and len(a.event_id) == 36
    assert abs((datetime.now() - a.timestamp).total_seconds()) < 5


def test_aggregate_root_events():
    g = Agg()
    assert g.domain_events == [] and g.id and g.created_at <= g.updated_at
    e1, e2 = DomainEvent(), DomainEvent()
    g.add_domain_event(e1)
    g.add_domain_event(e2)
    got = g.clear_domain_events()
    assert got == [e1, e2] and g.domain_events == []
    got.append(1)  # returned list is a copy, independent of internal state
    assert g.clear_domain_events() == []
    assert Agg().id != Agg().id


def test_abstract_classes_cannot_instantiate():
    for cls in (Repository, ApplicationService):
        with pytest.raises(TypeError):
            cls()


def test_abstract_bodies_are_noops_via_super():
    class R(Repository):
        def save(self, aggregate):
            return Repository.save(self, aggregate)

        def find_by_id(self, aggregate_id):
            return Repository.find_by_id(self, aggregate_id)

        def find_all(self):
            return Repository.find_all(self)

        def delete(self, aggregate_id):
            return Repository.delete(self, aggregate_id)

    class S(ApplicationService):
        def execute(self, *a, **k):
            return ApplicationService.execute(self, *a, **k)

    r = R()
    assert (r.save(None), r.find_by_id("x"), r.find_all(), r.delete("x")) == (None,) * 4
    assert S().execute(1, k=2) is None
    assert issubclass(DomainException, Exception)


def test_value_object_equality_hash():
    class V(ValueObject):
        def __init__(self, a, b):
            self.a, self.b = a, b

    class W(V):
        pass

    assert V(1, 2) == V(1, 2) and V(1, 2) != V(1, 3)
    assert V(1, 2) != W(1, 2) and V(1, 2) != (1, 2)
    assert hash(V(1, 2)) == hash(V(1, 2))
    assert len({V(1, 2), V(1, 2), V(2, 1)}) == 2


# ---------------- events ----------------
def test_event_classes_are_dataclasses_with_fields():
    now = datetime(2026, 1, 1)
    e = ev.MarketDataFetched("TCS", "NSE", now, {"c": 1.0}, {"pe": 2.0})
    assert (e.ticker, e.exchange, e.date) == ("TCS", "NSE", now)
    s = ev.ScreeningCompleted("darvas", ["A"], 10)
    assert s.timestamp is None and s.total_evaluated == 10
    assert ev.PortfolioRebalanced("p", {}, {}).timestamp is None
    assert ev.NotificationSent("r", "t", "s").timestamp is None
    n = ev.NotificationSent("r", "t", "s", now)
    assert n.timestamp == now


@pytest.mark.parametrize(
    "cls,args",
    [
        (ev.DataValidationFailed, ("T", datetime(2026, 1, 1), "bad")),
        (ev.StockScored, ("T", "s", 1.0, 0.5, ["a"], ["b"])),
        (ev.AllocationChanged, ("p", "T", 0.1, 0.2)),
        (ev.BacktestCompleted, ("p", datetime(2020, 1, 1), datetime(2021, 1, 1), 1, 2, 3, 4)),
        (ev.PerformanceReport, ("p", {})),
        (ev.RiskThresholdExceeded, ("p", "m", 1.0, 2.0, "HIGH")),
        (ev.BuySignalGenerated, ("T", 0.9, "r", 0.1)),
        (ev.SellSignalGenerated, ("T", 0.9, "r", 0.1)),
        (ev.ReportGenerated, ("daily", datetime(2026, 1, 1), "x")),
    ],
)
def test_event_construction_positional_order(cls, args):
    e = cls(*args)
    assert isinstance(e, DomainEvent) and dataclasses.is_dataclass(e)
    assert tuple(getattr(e, f.name) for f in dataclasses.fields(e))[: len(args)] == args


def test_event_dataclass_init_skips_base_init():
    # documents current behaviour: dataclass-generated __init__ does not call DomainEvent.__init__
    e = ev.PerformanceReport("p", {})
    assert not hasattr(e, "event_id") and not hasattr(e, "timestamp")


# ---------------- event bus ----------------
class Plain(DomainEvent):
    pass


@pytest.fixture
def bus():
    return eb.EventBus()


def test_publish_to_subscribers_and_history(bus, caplog):
    got = []

    def h1(e):
        got.append(("h1", e))

    def h2(e):
        got.append(("h2", e))

    bus.subscribe(Plain, h1)
    bus.subscribe(Plain, h2)
    assert bus.get_subscriber_count(Plain) == 2 and bus.get_subscriber_count(DomainEvent) == 0
    e = Plain()
    with caplog.at_level(logging.INFO):
        bus.publish(e)
    assert got == [("h1", e), ("h2", e)]  # subscription order
    assert bus.get_event_history() == [e]
    assert f"Publishing Plain (ID: {e.event_id})" in caplog.text


def test_publish_handler_failure_isolated(bus, caplog):
    got = []

    def bad(e):
        raise RuntimeError("boom")

    bus.subscribe(Plain, bad)
    bus.subscribe(Plain, lambda e: got.append(e))
    with caplog.at_level(logging.ERROR):
        bus.publish(Plain())
    assert len(got) == 1
    assert "Error in event handler bad: boom" in caplog.text


def test_publish_without_subscribers_warns_and_records(bus, caplog):
    with caplog.at_level(logging.WARNING):
        bus.publish(Plain())
    assert len(bus.get_event_history()) == 1
    assert "No subscribers for Plain" in caplog.text


def test_publish_is_exact_type_match(bus):
    class Sub(Plain):
        pass

    got = []
    bus.subscribe(Plain, got.append)
    bus.publish(Sub())
    assert got == []  # subclass events are not delivered to base-type subscribers


def test_history_copy_and_clear(bus):
    bus.publish(Plain())
    h = bus.get_event_history()
    h.clear()
    assert len(bus.get_event_history()) == 1
    bus.clear_history()
    assert bus.get_event_history() == []


def test_global_bus_singleton_and_reset(monkeypatch):
    monkeypatch.setattr(eb, "_event_bus", None)
    a = eb.get_event_bus()
    assert a is eb.get_event_bus()
    eb.reset_event_bus()
    assert eb.get_event_bus() is not a


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="event_bus.py:58 reads event.event_id, but @dataclass events in events.py never run DomainEvent.__init__, so publishing any of them raises",
)
def test_publish_dataclass_event_works(bus):
    got = []
    bus.subscribe(ev.PerformanceReport, got.append)
    e = ev.PerformanceReport("p", {})
    bus.publish(e)
    assert got == [e]


# ---------------- models ----------------
def test_ticker_validation():
    t = Ticker("TCS", "NSE")
    assert (
        str(t) == "TCS.NSE" and t == Ticker("TCS", "NSE") and hash(t) == hash(Ticker("TCS", "NSE"))
    )
    Ticker("ABCDEFGHIJ", "TSE")  # 10 chars ok
    with pytest.raises(ValueError, match="Invalid ticker"):
        Ticker("ABCDEFGHIJK", "NSE")
    with pytest.raises(ValueError, match="Invalid ticker"):
        Ticker("", "NSE")
    with pytest.raises(ValueError, match="Unknown exchange: LSE"):
        Ticker("X", "LSE")
    for ex in ["NSE", "BSE", "NASDAQ", "NYSE", "EURONEXT", "TSE"]:
        Ticker("X", ex)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.symbol = "Z"


def test_price():
    assert str(Price(Decimal("1.5"))) == "1.5 INR"
    assert str(Price(Decimal("0"), "USD")) == "0 USD"
    with pytest.raises(ValueError, match="negative"):
        Price(Decimal("-0.01"))


def _ohlc(o=2, h=3, l=1, c=2, v=0):
    return OHLC(Decimal(o), Decimal(h), Decimal(l), Decimal(c), v)


def test_ohlc_bounds():
    _ohlc(1, 3, 1, 3, 0)  # boundaries inclusive
    with pytest.raises(ValueError, match="Open"):
        _ohlc(o=4)
    with pytest.raises(ValueError, match="Open"):
        _ohlc(o=0)
    with pytest.raises(ValueError, match="Close"):
        _ohlc(c=4)
    with pytest.raises(ValueError, match="Close"):
        _ohlc(c=0)
    with pytest.raises(ValueError, match="Volume"):
        _ohlc(v=-1)


def test_metric():
    now = datetime(2026, 1, 1)
    assert Metric("PE", Decimal(1), now).name == "PE"
    with pytest.raises(ValueError, match="name required"):
        Metric("", Decimal(1), now)


def _md(date=None, metrics=None, sym="TCS"):
    return MarketData(Ticker(sym, "NSE"), date or datetime(2026, 1, 5), _ohlc(), metrics)


def test_market_data():
    m = _md()
    assert str(m) == "MarketData(TCS.NSE, 2026-01-05)" and m.metrics == {}
    with pytest.raises(ValueError, match="future"):
        _md(datetime.now() + timedelta(days=1))
    m.add_metric("PE", Decimal(1))
    assert not m.is_complete()
    m.add_metric("ROE", Decimal(1))
    m.add_metric("MarketCap", Decimal(1))
    assert m.is_complete()
    assert _md(metrics={"PE": 1, "ROE": 1, "MarketCap": 1}).is_complete()
    # metrics dict is not shared between instances
    assert _md().metrics is not _md().metrics


def test_market_data_stale_boundary():
    now = datetime.now()
    assert _md(now - timedelta(days=366, seconds=1)).is_stale()
    assert not _md(now - timedelta(days=365, seconds=1)).is_stale()  # exactly 365 days is not stale
    assert _md(now - timedelta(days=11)).is_stale(10)
    assert not _md(now - timedelta(days=10, hours=1)).is_stale(10)


def test_market_calendar():
    # Fri 2026-01-02 holiday-free; Mon 2026-01-05 holiday
    cal = MarketCalendar("NSE", {0, 1, 2, 3, 4}, {datetime(2026, 1, 5).date()})
    assert cal.is_trading_day(datetime(2026, 1, 2))
    assert not cal.is_trading_day(datetime(2026, 1, 3))  # Saturday
    assert not cal.is_trading_day(datetime(2026, 1, 5))  # holiday
    assert cal.is_trading_day(datetime(2026, 1, 6))
    assert cal.next_trading_day(datetime(2026, 1, 2)) == datetime(2026, 1, 6)
    assert cal.previous_trading_day(datetime(2026, 1, 6)) == datetime(2026, 1, 2)
    assert cal.next_trading_day(datetime(2026, 1, 6)) == datetime(2026, 1, 7)
    assert cal.previous_trading_day(datetime(2026, 1, 7)) == datetime(2026, 1, 6)


# ---------------- repositories ----------------
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="repositories.py:15 declares `class MarketDataRepository(ABC, Repository)`; Repository already derives from ABC so the MRO is inconsistent and the module cannot be imported",
)
def test_repositories_module_imports():
    import importlib
    import sys

    sys.modules.pop("src.market_data.domain.repositories", None)
    mod = importlib.import_module("src.market_data.domain.repositories")
    assert hasattr(mod, "MarketDataRepositoryMemory")


@pytest.fixture
def repos():
    """Load repositories.py with only the broken base-class list repaired (in memory;
    same filename/line numbers so coverage attributes to the real file)."""
    import pathlib
    import sys
    import types

    path = pathlib.Path(__file__).resolve().parent.parent / "src/market_data/domain/repositories.py"
    text = path.read_text()
    assert "class MarketDataRepository(ABC, Repository):" in text
    fixed = text.replace(
        "class MarketDataRepository(ABC, Repository):", "class MarketDataRepository(Repository):"
    )
    name = "src.market_data.domain._repos_fixed"
    mod = types.ModuleType(name)
    mod.__package__ = "src.market_data.domain"
    mod.__file__ = str(path)
    sys.modules[name] = mod
    try:
        exec(compile(fixed, str(path), "exec"), mod.__dict__)
        yield mod
    finally:
        sys.modules.pop(name, None)


def test_repository_abstract(repos):
    with pytest.raises(TypeError):
        repos.MarketDataRepository()


def test_repository_abstract_bodies(repos):
    class R(repos.MarketDataRepository):
        def save(self, m):
            return repos.MarketDataRepository.save(self, m)

        def find_by_ticker_and_date(self, t, d):
            return repos.MarketDataRepository.find_by_ticker_and_date(self, t, d)

        def find_by_ticker(self, t, days=252):
            return repos.MarketDataRepository.find_by_ticker(self, t, days)

        def find_all(self):
            return repos.MarketDataRepository.find_all(self)

        def delete(self, t, d=None):
            return repos.MarketDataRepository.delete(self, t, d)

        def find_by_id(self, i):
            return None

    r = R()
    t = Ticker("A", "NSE")
    assert [
        r.save(None),
        r.find_by_ticker_and_date(t, None),
        r.find_by_ticker(t),
        r.find_all(),
        r.delete(t, None),
    ] == [None] * 5


def test_memory_repository(repos):
    r = repos.MarketDataRepositoryMemory()
    t, u = Ticker("TCS", "NSE"), Ticker("INFY", "NSE")
    d1, d2, d3 = datetime(2026, 1, 5), datetime(2026, 1, 6, 15), datetime(2026, 1, 7)
    m1, m2, m3 = _md(d1), _md(d2), _md(d3)
    mu = _md(d1, sym="INFY")
    for m in (m1, m2, m3, mu):
        r.save(m)
    assert len(r.find_all()) == 4
    assert r.find_by_ticker_and_date(t, datetime(2026, 1, 6, 1)) is m2  # keyed by calendar date
    assert r.find_by_ticker_and_date(t, datetime(2026, 1, 9)) is None
    assert r.find_by_ticker(t) == [m3, m2, m1]  # newest first
    assert r.find_by_ticker(t, days=2) == [m3, m2]
    assert r.find_by_ticker(u) == [mu]
    assert r.find_by_ticker(Ticker("ZZZ", "NSE")) == []
    m1b = _md(d1)
    r.save(m1b)  # overwrite same key
    assert len(r.find_all()) == 4 and r.find_by_ticker_and_date(t, d1) is m1b
    r.delete(t, datetime(2026, 1, 5, 9))
    assert r.find_by_ticker_and_date(t, d1) is None
    r.delete(t, d1)  # deleting a missing key is a no-op
    assert len(r.find_all()) == 3
    with pytest.raises(NotImplementedError):
        r.find_by_id("x")
