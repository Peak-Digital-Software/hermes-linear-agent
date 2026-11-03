"""Customer-need mutations must select ``need`` (Linear has no ``customerNeed`` payload field)."""

from __future__ import annotations

from typing import Any

from hermes_linear_agent.client import LinearGraphQLClient


def _client(response: dict[str, Any], sent: list[str]) -> LinearGraphQLClient:
    client = LinearGraphQLClient("token")

    async def execute(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        sent.append(query)
        return response

    client.execute = execute  # type: ignore[method-assign]
    return client


async def test_create_customer_need_selects_need() -> None:
    sent: list[str] = []
    client = _client(
        {
            "customerNeedCreate": {
                "success": True,
                "need": {"id": "n1", "body": "b", "customer": {"id": "c1", "name": "Acme"}},
            }
        },
        sent,
    )
    result = await client.create_customer_need({"body": "b"}, mutation_policy={"create_customer_needs": True})
    assert "need {" in sent[0]
    assert "customerNeed {" not in sent[0]
    assert result["id"] == "n1"


async def test_update_customer_need_selects_need() -> None:
    sent: list[str] = []
    client = _client({"customerNeedUpdate": {"success": True, "need": {"id": "n1", "body": "b"}}}, sent)
    result = await client.update_customer_need("n1", {"body": "b"}, mutation_policy={"update_customer_needs": True})
    assert "need {" in sent[0]
    assert "customerNeed {" not in sent[0]
    assert result["id"] == "n1"
