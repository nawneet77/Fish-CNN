"""
FastAPI application for the Fish Health Dashboard.
Serves the dashboard HTML and provides WebSocket streaming.
"""

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="Fish Health Dashboard")

# Set by run_dashboard.py before startup
monitor_service = None


class ConnectionManager:
    """Manages WebSocket connections for live broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for conn in list(self.active_connections):
            try:
                await conn.send_json(data)
            except Exception:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(content=template_path.read_text())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "acknowledge_alert" and monitor_service:
                monitor_service.monitor.alert_system.acknowledge_alert(msg.get("alert_id", ""))
            elif msg.get("type") == "skip_baseline" and monitor_service:
                monitor_service.monitor.skip_baseline_collection()
            elif msg.get("type") == "tune" and monitor_service:
                monitor_service.apply_tuning(msg.get("param"), msg.get("value"))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/health-summary")
async def health_summary():
    if monitor_service and monitor_service.monitor:
        return monitor_service.monitor.get_health_summary()
    return {"error": "Monitor not running"}


@app.get("/api/alerts")
async def get_alerts():
    if monitor_service and monitor_service.monitor:
        alerts = monitor_service.monitor.alert_system.get_active_alerts()
        return [a.to_dict() for a in alerts]
    return []
