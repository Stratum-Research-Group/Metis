"""Prosopon upstream client package."""

from prosopon.client.kybernetes_client import KybernetesClient
from prosopon.client.mock_kybernetes import MockKybernetesClient

__all__ = ["KybernetesClient", "MockKybernetesClient"]
