"""Tiny trend bot for agent-fun/bars-replay/v1 (make.fun, pure stdlib Python).

Strategy (original): per-symbol dual-EMA trend following on hourly bars.
  - Fast EMA (8h) vs slow EMA (21h); go long only when the fast EMA sits
    above the slow EMA AND the latest close sits above the slow EMA, so the
    bot only holds assets in an established uptrend.
  - Skip a buy when the close is far above the slow EMA (overextended:
    more than 10% above it) to avoid chasing a spike.
  - Exit the whole position when the fast EMA crosses back below the slow
    EMA, or on a trailing stop: the close falls 5% below the highest close
    seen since entry.
  - Volatility-aware sizing: the target weight per symbol is scaled down
    when recent hourly closes are jumpy (measured from close-to-close
    moves over the last 24 bars), never above 15% of equity and never
    above the harness max-weight cap. Small trades are skipped so fees
    and slippage do not eat the edge; orders are only placed on signal
    flips, never every bar.
  - Long only, no leverage, no shorts. Deterministic: no clock, no
    network, no randomness, no file reads. The bot reads each hourly bar
    batch, answers with exactly one reply echoing its seq, and the orders
    in that reply fill at the next bar's open.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

FAST_SPAN = 8
SLOW_SPAN = 21
VOL_WINDOW = 24
MAX_WEIGHT = 0.15          # at most 15% of equity in one symbol
MIN_TRADE_USDC = 25        # skip dust-sized trades (units are micro-USDC)
OVEREXTEND = 0.10          # do not chase a close >10% above the slow EMA
STOP_DROP = 0.05           # trailing stop: 5% below the peak close since entry
MAX_SLIPPAGE_BPS = 200

# Per-symbol state, kept across batches.
closes = {}    # symbol -> list of recent closes (floats, USDC)
peak = {}      # symbol -> highest close seen while holding it


def _ema(prev, price, span):
    k = 2.0 / (span + 1.0)
    return price if prev is None else price * k + prev * (1.0 - k)


def _ema_series(values, span):
    out = None
    for v in values:
        out = _ema(out, v, span)
    return out


def _volatility(values):
    """Mean absolute hourly close-to-close move, as a fraction of price."""
    if len(values) < 3:
        return 0.0
    total = 0.0
    n = 0
    for a, b in zip(values, values[1:]):
        if a > 0:
            total += abs(b - a) / a
            n += 1
    return total / n if n else 0.0


def on_batch(state, batch):
    # Record closes even during warm-up batches (orders on those are
    # rejected, but the history is what lets the bot trade from the first
    # scored bar instead of waiting another slow-EMA span).
    for bar in batch["bars"]:
        sym = bar["symbol"]
        info = state.symbols.get(sym)
        if info is None:
            continue
        close = int(bar["c"])
        if close <= 0:
            continue
        hist = closes.setdefault(sym, [])
        hist.append(close / (10 ** info["price_scale"]))
        if len(hist) > 120:
            del hist[:-120]
    if state.warming_up:
        return []
    orders = []
    cash = state.cash
    equity = state.equity if state.equity > 0 else state.cash
    for bar in batch["bars"]:
        sym = bar["symbol"]
        close = int(bar["c"])
        info = state.symbols.get(sym)
        if info is None or close <= 0:
            continue
        price = close / (10 ** info["price_scale"])
        hist = closes.get(sym, [])
        if len(hist) < SLOW_SPAN + 2:
            continue
        fast = _ema_series(hist, FAST_SPAN)
        slow = _ema_series(hist, SLOW_SPAN)
        held = state.positions.get(sym)
        holding = held is not None and held.qty > 0
        if holding:
            if price > peak.get(sym, 0.0):
                peak[sym] = price
        # Exit signals.
        exit_now = False
        if holding:
            if fast < slow:
                exit_now = True
            elif peak.get(sym, 0.0) > 0 and price < peak[sym] * (1.0 - STOP_DROP):
                exit_now = True
        if exit_now:
            orders.append({
                "side": "sell",
                "symbol": sym,
                "qty": held.qty,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            })
            peak.pop(sym, None)
            continue
        # Entry signals.
        if holding:
            continue
        if fast <= slow or price <= slow:
            continue
        if price > slow * (1.0 + OVEREXTEND):
            continue
        vol = _volatility(hist[-VOL_WINDOW:])
        # Calmer assets get a bigger slice; jumpy ones a smaller one.
        scale = 1.0
        if vol > 0.02:
            scale = 0.6
        elif vol > 0.01:
            scale = 0.8
        target = int(equity * MAX_WEIGHT * scale)
        room = state.weight_room(sym)
        if room < target:
            target = room
        if target < MIN_TRADE_USDC * 1_000_000:
            continue
        if target > cash:
            target = cash
        if target < MIN_TRADE_USDC * 1_000_000:
            continue
        # Leave a little cash headroom when several symbols fire at once.
        spend = target
        if spend > cash:
            spend = cash
        if spend <= 0:
            continue
        cash -= spend
        peak[sym] = price
        orders.append({
            "side": "buy",
            "symbol": sym,
            "notional_quote_units": spend,
            "max_slippage_bps": MAX_SLIPPAGE_BPS,
        })
        if len(orders) >= state.hello["caps"]["max_orders_per_bar"]:
            break
    return orders


makefun_bars.run(on_batch)
