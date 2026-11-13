"""In-app notifications system for warranty alerts, lifecycle events, and safety warnings."""

from __future__ import annotations

from typing import Any


class NotificationChannel:
    """Base notification channel interface for modular future expansion."""

    def dispatch(self, notifications: list[dict[str, Any]]) -> None:
        raise NotImplementedError


class InAppNotificationChannel(NotificationChannel):
    """In-app notification channel rendering directly into Streamlit UI."""

    def dispatch(self, notifications: list[dict[str, Any]]) -> None:
        # In-app notifications are directly queried and rendered by Streamlit views
        pass


def collect_battery_notifications(lifecycle_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gather and categorize actionable notifications across all registered batteries.

    Returns notifications sorted by urgency:
      1. Critical safety / abnormal behaviour
      2. Expected lifespan exceeded
      3. Poor battery health
      4. Warranty expiring soon
      5. Warranty expired
    """
    notifications: list[dict[str, Any]] = []

    for summary in lifecycle_summaries:
        identifier = summary.get("battery_identifier", "Unknown")
        warranty = summary.get("warranty") or {}
        lifespan = summary.get("lifespan") or {}
        rec = summary.get("recommendation") or {}
        latest_soh = summary.get("latest_soh")

        # 1. Critical safety notification
        if rec.get("level") == "CRITICAL":
            notifications.append(
                {
                    "battery_identifier": identifier,
                    "category": "critical_safety",
                    "severity": "error",
                    "priority": 1,
                    "title": f"Critical Safety Alert: {identifier}",
                    "message": rec["recommendation"],
                    "action": rec.get("action", "Inspect"),
                }
            )

        # 2. Expected lifespan exceeded notification
        if lifespan.get("is_exceeded"):
            notifications.append(
                {
                    "battery_identifier": identifier,
                    "category": "lifespan_exceeded",
                    "severity": "error",
                    "priority": 2,
                    "title": f"Lifespan Exceeded: {identifier}",
                    "message": (
                        "Your battery has exceeded its expected lifespan. "
                        "We recommend arranging a professional battery inspection and considering battery replacement."
                    ),
                    "action": "Arrange Replacement",
                }
            )

        # 3. Poor battery health notification
        elif rec.get("level") == "POOR_HEALTH":
            soh_str = f" ({latest_soh:.1f}% SoH)" if latest_soh is not None else ""
            notifications.append(
                {
                    "battery_identifier": identifier,
                    "category": "poor_health",
                    "severity": "error",
                    "priority": 3,
                    "title": f"Significant Degradation: {identifier}{soh_str}",
                    "message": (
                        "Your battery health is significantly degraded. "
                        "We recommend professional inspection and considering battery replacement."
                    ),
                    "action": "Inspect / Replace",
                }
            )

        # 4. Warranty Expiring Soon
        if warranty.get("is_expiring_soon"):
            days = warranty.get("days_remaining", 0)
            notifications.append(
                {
                    "battery_identifier": identifier,
                    "category": "warranty_expiring_soon",
                    "severity": "warning",
                    "priority": 4,
                    "title": f"Warranty Expiring Soon: {identifier}",
                    "message": (
                        f"Your battery warranty will expire in {days} day{'s' if days != 1 else ''}. "
                        "Please contact your battery manufacturer or service provider."
                    ),
                    "action": "Contact Manufacturer",
                }
            )

        # 5. Warranty Expired (Separate notice; not confused with replacement EOL)
        elif warranty.get("is_expired"):
            notifications.append(
                {
                    "battery_identifier": identifier,
                    "category": "warranty_expired",
                    "severity": "info",
                    "priority": 5,
                    "title": f"Warranty Expired: {identifier}",
                    "message": (
                        "Your battery warranty has expired. Check your battery's condition and available service options. "
                        "(Note: Warranty expiry does not automatically mean that the battery needs replacement.)"
                    ),
                    "action": "Review Coverage",
                }
            )

    # Sort notifications by priority ascending (1 = most urgent)
    notifications.sort(key=lambda x: x["priority"])
    return notifications

