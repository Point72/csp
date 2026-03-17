"""
FastAPI with CSP Event Loop Integration

This example demonstrates two ways to integrate CSP with FastAPI:

1. Using CSP's asyncio bridge to push data from FastAPI to a CSP graph
2. Running CSP graphs that feed data to FastAPI endpoints

Note: CSP's event loop doesn't yet implement create_server(), so we can't
replace uvicorn's event loop entirely. Instead, we show how to bridge
asyncio web frameworks with CSP graphs running in realtime mode.

To run this example:
    pip install fastapi uvicorn
    python e2_csp_fastapi.py

Then visit (check console output for actual port):
    http://localhost:<port>/        - Hello world
    http://localhost:<port>/docs    - Swagger UI
    http://localhost:<port>/price   - Real-time price from CSP graph
    http://localhost:<port>/submit  - Submit data to CSP graph
"""

import random
import socket
import time
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Optional

# Check for FastAPI/uvicorn availability
try:
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    print("This example requires FastAPI and uvicorn.")
    print("Install with: pip install fastapi uvicorn")
    exit(1)

import csp
from csp import ts
from csp.event_loop import BidirectionalBridge
from csp.utils.datetime import utc_now

# Utilities


def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")


# Latest values from CSP graph (updated by CSP, read by FastAPI)
_latest_prices: Dict[str, dict] = {}
_prices_lock = Lock()

# Bridge for sending data from FastAPI to CSP
_to_csp_bridge: Optional[BidirectionalBridge] = None
_csp_runner = None


@csp.node
def generate_prices(trigger: ts[bool], symbols: list, bridge: object) -> ts[dict]:
    """Generate simulated market prices and emit to FastAPI."""
    with csp.state():
        s_prices = {}

    if csp.ticked(trigger):
        for sym in symbols:
            if sym not in s_prices:
                s_prices[sym] = 100.0

            # Random walk
            change = random.uniform(-0.5, 0.5)
            s_prices[sym] = max(0.01, s_prices[sym] + change)

            price_data = {
                "symbol": sym,
                "price": round(s_prices[sym], 2),
                "timestamp": str(csp.now()),
            }

            # Emit to FastAPI via bridge
            if bridge is not None:
                bridge.emit(price_data)

        return {"prices": s_prices.copy(), "time": str(csp.now())}


@csp.node
def process_orders(
    orders: ts[dict],
) -> ts[dict]:
    """Process orders received from FastAPI."""
    if csp.ticked(orders):
        order = orders
        result = {
            "order_id": order.get("id"),
            "symbol": order.get("symbol"),
            "quantity": order.get("quantity"),
            "status": "FILLED",
            "fill_price": round(random.uniform(99, 101), 2),
            "processed_at": str(csp.now()),
        }
        print(f"  CSP processed order: {result}")
        return result


@csp.graph
def market_data_graph(
    symbols: list,
    price_bridge: object,
    order_bridge: object,
):
    """Graph that generates prices and processes orders."""
    # Generate prices every 500ms
    timer = csp.timer(timedelta(milliseconds=500))
    prices = generate_prices(timer, symbols, price_bridge)

    # Process orders from FastAPI
    if order_bridge is not None:
        orders = order_bridge.adapter.out()
        results = process_orders(orders)
        csp.add_graph_output("fills", results)

    csp.add_graph_output("prices", prices)


app = FastAPI(
    title="CSP + FastAPI Integration",
    description="FastAPI app integrated with a CSP graph for real-time data",
    version="1.0.0",
)


class Order(BaseModel):
    """Order request model."""

    symbol: str
    quantity: int
    side: str = "BUY"


@app.get("/")
async def root():
    """Hello world endpoint."""
    return {
        "message": "FastAPI + CSP Integration Example",
        "description": "This app runs FastAPI alongside a CSP graph",
        "csp_running": _csp_runner is not None and _csp_runner.is_alive(),
    }


@app.get("/prices")
async def get_prices():
    """Get latest prices from CSP graph."""
    with _prices_lock:
        return {
            "prices": dict(_latest_prices),
            "source": "CSP real-time graph",
        }


@app.get("/price/{symbol}")
async def get_price(symbol: str):
    """Get price for a specific symbol."""
    symbol = symbol.upper()
    with _prices_lock:
        if symbol in _latest_prices:
            return _latest_prices[symbol]
        return {"error": f"Symbol {symbol} not found", "available": list(_latest_prices.keys())}


@app.post("/order")
async def submit_order(order: Order):
    """Submit an order to CSP for processing."""
    if _to_csp_bridge is None:
        return {"error": "CSP graph not running"}

    order_data = {
        "id": f"ORD-{int(time.time() * 1000)}",
        "symbol": order.symbol.upper(),
        "quantity": order.quantity,
        "side": order.side,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    # Push to CSP graph via bridge
    _to_csp_bridge.push(order_data)

    return {
        "status": "submitted",
        "order": order_data,
        "note": "Order sent to CSP for processing",
    }


@app.get("/status")
async def status():
    """Get system status."""
    return {
        "fastapi": "running",
        "csp_graph": "running" if _csp_runner and _csp_runner.is_alive() else "stopped",
        "symbols_tracked": list(_latest_prices.keys()),
        "server_time": datetime.utcnow().isoformat(),
    }


def on_price_update(price_data: dict):
    """Callback when CSP emits a price update."""
    symbol = price_data.get("symbol")
    if symbol:
        with _prices_lock:
            _latest_prices[symbol] = price_data


def start_csp_graph():
    """Start the CSP graph in a background thread."""
    global _to_csp_bridge, _csp_runner

    # Create bridges
    price_bridge = BidirectionalBridge(dict, "prices")
    _to_csp_bridge = BidirectionalBridge(dict, "orders")

    # Register callback for price updates
    price_bridge.on_event(on_price_update)

    # Start bridges
    price_bridge.start()
    _to_csp_bridge.start()

    # Define symbols to track
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    # Run CSP graph in background thread
    start_time = utc_now()
    _csp_runner = csp.run_on_thread(
        market_data_graph,
        symbols,
        price_bridge,
        _to_csp_bridge,
        realtime=True,
        starttime=start_time,
        endtime=timedelta(hours=1),  # Run for 1 hour
    )

    print(f"CSP graph started, tracking: {symbols}")
    return price_bridge, _to_csp_bridge


def main():
    """Run the FastAPI app with CSP integration."""
    print("=" * 60)
    print("FastAPI + CSP Integration Example")
    print("=" * 60)
    print()

    # Start CSP graph
    print("Starting CSP graph...")
    price_bridge, order_bridge = start_csp_graph()
    time.sleep(0.5)  # Let CSP start

    # Find an available port
    port = find_free_port()

    print(f"Starting FastAPI server on http://localhost:{port}")
    print()
    print("Endpoints:")
    print("  GET  /           - Status")
    print("  GET  /prices     - All latest prices from CSP")
    print("  GET  /price/AAPL - Price for specific symbol")
    print("  POST /order      - Submit order to CSP")
    print("  GET  /status     - System status")
    print("  GET  /docs       - Swagger UI")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
    finally:
        print("\nStopping CSP graph...")
        price_bridge.stop()
        order_bridge.stop()


if __name__ == "__main__":
    main()
