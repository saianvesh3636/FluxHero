"""
API Routes Package

Contains modular route definitions for the FluxHero API.
"""

from backend.api.routes.reports import router as reports_router

__all__ = ["reports_router"]
