"""
Health Check Router
Comprehensive health check endpoints for system monitoring
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime

from src.shared.config import get_settings
from src.shared.health import health_monitor, HealthStatus

router = APIRouter()


class ComponentHealth(BaseModel):
    component: str
    status: str
    message: str
    details: Dict[str, Any]
    response_time: float
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: List[ComponentHealth]


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    settings = get_settings()

    # Perform system health check
    system_health = await health_monitor.check_all()

    # Convert to response format
    components = [
        ComponentHealth(
            component=result.component,
            status=result.status.value,
            message=result.message,
            details=result.details,
            response_time=result.response_time,
            timestamp=result.timestamp,
        )
        for result in system_health.components
    ]

    return HealthResponse(
        status=system_health.overall_status.value,
        timestamp=system_health.timestamp,
        version=settings.version,
        uptime=system_health.uptime,
        components=components,
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes/Docker"""
    system_health = await health_monitor.check_all()

    # System is ready if overall status is healthy or degraded
    if system_health.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
        return {"status": "ready", "timestamp": datetime.utcnow()}
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "overall_status": system_health.overall_status.value,
                "timestamp": datetime.utcnow(),
            },
        )


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes/Docker"""
    # Simple liveness check - if we can respond, we're alive
    return {
        "status": "alive",
        "timestamp": datetime.utcnow(),
        "uptime": health_monitor.get_uptime(),
    }


@router.get("/components")
async def list_components():
    """List all monitored components"""
    components = health_monitor.get_component_names()
    last_results = health_monitor.get_last_results()

    component_info = []
    for component in components:
        info = {"name": component}
        if component in last_results:
            result = last_results[component]
            info.update(
                {
                    "last_status": result.status.value,
                    "last_check": result.timestamp,
                    "last_response_time": result.response_time,
                }
            )
        component_info.append(info)

    return {"components": component_info, "total_count": len(components)}


@router.get("/components/{component_name}")
async def check_component(component_name: str):
    """Check health of a specific component"""
    result = await health_monitor.check_component(component_name)

    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Component '{component_name}' not found"
        )

    return ComponentHealth(
        component=result.component,
        status=result.status.value,
        message=result.message,
        details=result.details,
        response_time=result.response_time,
        timestamp=result.timestamp,
    )
