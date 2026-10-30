"""Linear Agent Session platform adapter for Hermes."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from utils import is_truthy_value as _boolish

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency checked at registration
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    resolve_proxy_url,
)
from gateway.platforms.helpers import MessageDeduplicator

# Importing .tools registers every linear_agent_* tool (and the toolset
# alias) as an import side effect.
try:
    from . import tools  # noqa: F401 - side-effect import triggers registration
except Exception as e:
    logging.getLogger(__name__).warning(f"[linear_agent] Failed to register tools: {e}")

from .client import DEFAULT_LINEAR_GRAPHQL_URL, LinearGraphQLClient
from .oauth import (
    LINEAR_TOKEN_URL,
    LinearOAuthConfig,
    LinearOAuthTokenManager,
    build_auth_token_update_callback,
    read_auth_token,
)
from .registry import set_active_adapter
from .webhook import (
    SUPPORTED_ACTIONS,
    LinearWebhookContext,
    LinearWebhookError,
    build_created_prompt,
    build_prompted_message,
    build_update_prompt,
    describe_signature_headers,
    extract_context,
    is_authorized,
    is_stale_body_timestamp,
    normalize_action,
    parse_json_body,
    verify_linear_signature,
)

logger = logging.getLogger(__name__)

# Loopback by default: the webhook listener is reachable from the public
# internet through a tunnel/reverse proxy the operator controls, not by
# binding every interface. Set `webhook_host` (or LINEAR_AGENT_WEBHOOK_HOST)
# to widen it deliberately.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8651
DEFAULT_WEBHOOK_PATH = "/hermes/linear-agent"
DEFAULT_ACK_BODY = "I’m starting work on this."
# Agent-to-agent handoff cap. A peer session is accepted only while the issue
# still has at most this many app-initiated sessions; beyond it a human has to
# take over (two agents can otherwise ping an issue back and forth forever).
DEFAULT_MAX_AGENT_HANDOFFS = 2
# All write operations fail closed: each key must be explicitly enabled.
# `update_projects` also covers project updates (status updates), milestones,
# and initiatives — project-structure mutations share one umbrella key.
DEFAULT_MUTATION_POLICY = {
    "create_comments": False,
    "update_comments": False,
    "update_issues": False,
    "create_issues": False,
    "update_projects": False,
    "create_documents": False,
    "update_documents": False,
    "create_customer_needs": False,
    "update_customer_needs": False,
    # `create_releases`/`update_releases` also gate release NOTES (the
    # release-family umbrella, same precedent as milestones/status updates
    # under `update_projects`).
    "create_releases": False,
    "update_releases": False,
    "create_customers": False,
    "update_customers": False,
    "create_labels": False,
    # Deletes are their own fail-closed family, separate from create/update.
    "delete_comments": False,
    "delete_customer_needs": False,
    "delete_status_updates": False,
    "delete_attachments": False,
    "delete_customers": False,
}
_DEDUP_TTL_SECONDS = 3600
MAX_MESSAGE_LENGTH = 40_000


def _str_env(name: str) -> str:
    """Profile-scoped env read: under gateway.multiplex_profiles a profile's .env
    lives only in its secret scope, so os.getenv would miss it (or read the
    default profile's value)."""
    try:
        from gateway.platforms._shared import get_scoped_secret
    except ImportError:  # older Hermes without secret scopes
        return os.getenv(name, "").strip()
    return str(get_scoped_secret(name, "", external_fallback=True) or "").strip()


def _first_secret(extra: dict[str, Any], key: str, env_name: str) -> str:
    return str(extra.get(key) or _str_env(env_name) or "").strip()


def _bool_opt(extra: dict[str, Any], key: str, env_name: str, default: bool) -> bool:
    """Tri-state boolean option: YAML wins, then env var, then the default.

    An explicit ``false`` in YAML must survive — an ``or`` chain would
    collapse it into the env/default fallback and silently re-enable the
    feature.
    """
    value = extra.get(key)
    if value is None:
        value = _str_env(env_name) or None
    return _boolish(value, default)


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _intish(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_auth_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "plugin-state" / "linear-agent" / "oauth.json"
    except Exception:
        return Path.home() / ".hermes" / "plugin-state" / "linear-agent" / "oauth.json"


def _legacy_auth_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "auth.json"
    except Exception:
        return Path.home() / ".hermes" / "auth.json"


def _auth_path(extra: dict[str, Any]) -> Path:
    return Path(str(extra.get("auth_path") or _str_env("LINEAR_AGENT_AUTH_PATH") or _default_auth_path()))


def _cached_auth_token(extra: dict[str, Any]) -> dict[str, Any]:
    state = read_auth_token(_auth_path(extra))
    if state:
        return state
    # One-way migration compatibility: older in-tree versions stored the
    # Linear token under providers.linear_agent in Hermes' shared auth.json.
    # Read it only when no explicit plugin state path was configured; all new
    # writes go to the plugin-owned path above.
    if not extra.get("auth_path") and not _str_env("LINEAR_AGENT_AUTH_PATH"):
        return read_auth_token(_legacy_auth_path())
    return {}


def _cached_access_token(extra: dict[str, Any]) -> str:
    return str(_cached_auth_token(extra).get("access_token") or "").strip()


def _oauth_token_config(extra: dict[str, Any]) -> bool:
    """Return True when OAuth credentials can issue/refresh access tokens."""
    return bool(
        _first_secret(extra, "client_id", "LINEAR_AGENT_CLIENT_ID")
        and _first_secret(extra, "client_secret", "LINEAR_AGENT_CLIENT_SECRET")
    )


def _build_oauth_manager(extra: dict[str, Any], access_token: str) -> LinearOAuthTokenManager | None:
    client_id = _first_secret(extra, "client_id", "LINEAR_AGENT_CLIENT_ID")
    client_secret = _first_secret(extra, "client_secret", "LINEAR_AGENT_CLIENT_SECRET")
    auth_path = _auth_path(extra)
    auth_state = _cached_auth_token(extra)
    refresh_token = (
        _first_secret(extra, "refresh_token", "LINEAR_AGENT_REFRESH_TOKEN")
        or str(auth_state.get("refresh_token") or "").strip()
    )
    access_token = access_token or str(auth_state.get("access_token") or "").strip()
    if not (client_id and client_secret):
        return None
    expires_at = _floatish(
        extra.get("token_expires_at") or _str_env("LINEAR_AGENT_TOKEN_EXPIRES_AT") or auth_state.get("expires_at")
    )
    token_url = str(
        extra.get("token_url") or _str_env("LINEAR_AGENT_TOKEN_URL") or auth_state.get("token_url") or LINEAR_TOKEN_URL
    )
    oauth_scopes = str(
        extra.get("oauth_scopes") or _str_env("LINEAR_AGENT_OAUTH_SCOPES") or auth_state.get("scope") or "read,write"
    )
    persist_tokens = _bool_opt(extra, "persist_tokens", "LINEAR_AGENT_PERSIST_TOKENS", True)
    if persist_tokens:
        callback = build_auth_token_update_callback(
            auth_path,
            client_id=client_id,
            token_url=token_url,
            scope=oauth_scopes,
        )
    else:
        callback = None
    return LinearOAuthTokenManager(
        LinearOAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            access_token=access_token,
            expires_at=expires_at,
            token_url=token_url,
            oauth_scopes=oauth_scopes,
            persist_callback=callback,
        )
    )


def _normalize_path(value: Any, default: str = DEFAULT_WEBHOOK_PATH) -> str:
    path = str(value or default).strip() or default
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


class LinearAgentConfigError(ValueError):
    """Invalid Linear Agent config; raised at adapter construction (fail closed)."""


# The only string spellings a boolean mutation_policy value may use. Anything
# else raises instead of being coerced: a typo'd policy that silently became
# True would grant a write the operator never enabled.
_POLICY_TRUE = frozenset({"true", "yes", "1"})
_POLICY_FALSE = frozenset({"false", "no", "0"})


def _policy_flag(key: str, value: Any) -> bool:
    """Coerce one ``mutation_policy`` value strictly (bool, or true/false/yes/no/1/0)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _POLICY_TRUE:
            return True
        if lowered in _POLICY_FALSE:
            return False
    raise LinearAgentConfigError(
        f"[linear_agent] mutation_policy.{key} must be a boolean (true/false/yes/no/1/0), got {value!r}"
    )


def _merge_policy(value: Any) -> dict[str, Any]:
    policy = dict(DEFAULT_MUTATION_POLICY)
    if isinstance(value, dict):
        unknown = set(value) - set(DEFAULT_MUTATION_POLICY)
        if unknown:
            # A typo (e.g. create_comment, singular) fails closed but would
            # otherwise silently ignore the operator's intent.
            logger.warning(
                "[linear_agent] Unknown mutation_policy keys ignored: %s",
                ", ".join(sorted(unknown)),
            )
        # Only known keys are copied, and only after strict coercion: an
        # unknown key can never smuggle a value into the effective policy.
        for key in DEFAULT_MUTATION_POLICY:
            if key in value:
                policy[key] = _policy_flag(key, value[key])
    return policy


def _split_message(content: str, limit: int) -> list[str]:
    """Split on paragraph, then line, then hard boundaries to fit ``limit``."""
    if len(content) <= limit:
        return [content]
    chunks: list[str] = []
    remaining = content
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def _resolve_linear_proxy(extra: dict[str, Any], api_url: str) -> str | None:
    """Resolve an outbound proxy for Linear API traffic.

    ``proxy_url`` in config wins; otherwise the shared resolver checks
    LINEAR_AGENT_PROXY, the standard *_PROXY env vars, and the macOS system
    proxy — honoring NO_PROXY for the API host.
    """
    configured = str(extra.get("proxy_url") or "").strip()
    if configured:
        return configured
    host = urlparse(api_url).hostname or "api.linear.app"
    return resolve_proxy_url("LINEAR_AGENT_PROXY", target_hosts=host)


def check_requirements() -> bool:
    """Return True when the adapter's runtime dependency is available."""
    return AIOHTTP_AVAILABLE


def validate_config(config: PlatformConfig) -> bool:
    """Validate the minimum config required to run Linear Agent sessions."""
    extra = config.extra or {}
    token = (
        config.token
        or extra.get("access_token")
        or _cached_access_token(extra)
        or _str_env("LINEAR_AGENT_ACCESS_TOKEN")
    )
    return bool(str(token or "").strip() or _oauth_token_config(extra))


def is_connected(config: PlatformConfig) -> bool:
    return validate_config(config)


def _env_enablement() -> dict[str, Any] | None:
    token = _str_env("LINEAR_AGENT_ACCESS_TOKEN")
    seed: dict[str, Any] = {}
    cached_token = _cached_access_token({})
    if cached_token:
        seed["access_token"] = cached_token
    elif token:
        seed["access_token"] = token
    for env_name, key in (
        ("LINEAR_AGENT_WEBHOOK_SECRET", "webhook_secret"),
        ("LINEAR_AGENT_APP_USER_ID", "app_user_id"),
        ("LINEAR_AGENT_WORKSPACE_ID", "workspace_id"),
        ("LINEAR_AGENT_CLIENT_ID", "client_id"),
        ("LINEAR_AGENT_CLIENT_SECRET", "client_secret"),
        ("LINEAR_AGENT_REFRESH_TOKEN", "refresh_token"),
        ("LINEAR_AGENT_TOKEN_EXPIRES_AT", "token_expires_at"),
        ("LINEAR_AGENT_REDIRECT_URI", "redirect_uri"),
        ("LINEAR_AGENT_OAUTH_SCOPES", "oauth_scopes"),
        ("LINEAR_AGENT_HOME_TARGET", "home_target"),
        ("LINEAR_AGENT_ALLOWED_USERS", "allowed_users"),
        ("LINEAR_AGENT_PEER_APP_USER_IDS", "peer_app_user_ids"),
        ("LINEAR_AGENT_MAX_AGENT_HANDOFFS", "max_agent_handoffs"),
        ("LINEAR_AGENT_ALLOWED_TEAMS", "allowed_teams"),
        ("LINEAR_AGENT_ALLOW_ALL_USERS", "allow_all_users"),
    ):
        value = _str_env(env_name)
        if value:
            seed[key] = value
    return seed or None


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict[str, Any] | None:
    """Bridge `linear_agent:` config into PlatformConfig.extra."""
    seed: dict[str, Any] = {}
    for key in (
        "webhook_host",
        "host",
        "webhook_port",
        "port",
        "webhook_path",
        "workspace_id",
        "app_user_id",
        "client_id",
        "client_secret",
        "allowed_teams",
        "allowed_users",
        "peer_app_user_ids",
        "max_agent_handoffs",
        "allow_all_users",
        "ack_on_created",
        "ack_body",
        "auto_start_on_delegation",
        "auto_self_delegate",
        "dispatch_issue_updates",
        "reply_in_source_thread",
        "allow_unsigned_webhooks",
        "proxy_url",
        "mutation_policy",
        "auto_skills",
        "api_url",
        "token_url",
        "redirect_uri",
        "oauth_scopes",
        "refresh_token",
        "token_expires_at",
        "persist_tokens",
        "env_path",
        "auth_path",
        "max_body_bytes",
    ):
        if key in platform_cfg:
            seed[key] = platform_cfg[key]

    extra = platform_cfg.get("extra")
    if isinstance(extra, dict):
        seed.update(extra)

    if platform_cfg.get("access_token") and not _str_env("LINEAR_AGENT_ACCESS_TOKEN"):
        seed["access_token"] = platform_cfg["access_token"]
    if platform_cfg.get("client_id") and not _str_env("LINEAR_AGENT_CLIENT_ID"):
        seed["client_id"] = platform_cfg["client_id"]
    if platform_cfg.get("client_secret") and not _str_env("LINEAR_AGENT_CLIENT_SECRET"):
        seed["client_secret"] = platform_cfg["client_secret"]
    if platform_cfg.get("webhook_secret") and not _str_env("LINEAR_AGENT_WEBHOOK_SECRET"):
        seed["webhook_secret"] = platform_cfg["webhook_secret"]

    return seed or None


def _resolve_access_token(config_token: Any, extra: dict[str, Any]) -> str:
    """Credential chain shared by adapter startup and standalone cron sends:
    config token, YAML access_token, cached OAuth token, then env.

    The winning source is logged because a stale cached plugin-state token
    silently shadowing a freshly rotated env token is otherwise
    undiagnosable (every send 401s while the env token looks correct).
    """
    for source, value in (
        ("config token", config_token),
        ("yaml access_token", extra.get("access_token")),
        ("cached plugin-state token", _cached_access_token(extra)),
        ("LINEAR_AGENT_ACCESS_TOKEN env", _str_env("LINEAR_AGENT_ACCESS_TOKEN")),
    ):
        token = str(value or "").strip()
        if token:
            logger.debug("[linear_agent] Access token source: %s", source)
            return token
    return ""


def _build_client(extra: dict[str, Any], access_token: str, token_manager) -> LinearGraphQLClient:
    api_url = str(extra.get("api_url") or DEFAULT_LINEAR_GRAPHQL_URL)
    return LinearGraphQLClient(
        access_token,
        api_url=api_url,
        token_manager=token_manager,
        proxy_url=_resolve_linear_proxy(extra, api_url),
    )


class LinearAgentAdapter(BasePlatformAdapter):
    """Receive Linear Agent Session webhooks and reply with Agent Activities."""

    supports_code_blocks = True
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    # Agent Activities are immutable — without edit support, streaming would
    # stack partial `response` activities on the session timeline.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(
        self,
        config: PlatformConfig,
        *,
        client: LinearGraphQLClient | None = None,
    ) -> None:
        super().__init__(config=config, platform=Platform("linear_agent"))
        extra = config.extra or {}
        self._host = str(extra.get("webhook_host") or extra.get("host") or DEFAULT_HOST)
        # None-checked (not `or`-chained): an explicit 0 means "bind an
        # ephemeral port" and must not collapse into the default.
        port_val = extra.get("webhook_port")
        if port_val in (None, ""):
            port_val = extra.get("port")
        self._port = DEFAULT_PORT if port_val in (None, "") else int(port_val)
        self._webhook_path = _normalize_path(extra.get("webhook_path"))
        self._access_token = _resolve_access_token(config.token, extra)
        self._webhook_secret = str(extra.get("webhook_secret") or _str_env("LINEAR_AGENT_WEBHOOK_SECRET") or "").strip()
        self._allow_unsigned_webhooks = _bool_opt(
            extra,
            "allow_unsigned_webhooks",
            "LINEAR_AGENT_ALLOW_UNSIGNED_WEBHOOKS",
            False,
        )
        self._workspace_id = str(extra.get("workspace_id") or _str_env("LINEAR_AGENT_WORKSPACE_ID") or "").strip()
        self._app_user_id = str(extra.get("app_user_id") or _str_env("LINEAR_AGENT_APP_USER_ID") or "").strip()
        # YAML wins; LINEAR_AGENT_* env vars are fallbacks. The gateway auth
        # layer reads the same env vars independently.
        self._allowed_teams = _list_value(extra.get("allowed_teams") or _str_env("LINEAR_AGENT_ALLOWED_TEAMS"))
        self._allowed_users = _list_value(extra.get("allowed_users") or _str_env("LINEAR_AGENT_ALLOWED_USERS"))
        # Other agent apps allowed to hand work to this one. A peer session
        # arrives with no human creator (Linear sets AgentSession.creator =
        # null), so without this list every machine handoff would look like an
        # unauthorized sender. Never includes our own app user id.
        self._peer_app_user_ids = _list_value(
            extra.get("peer_app_user_ids") or _str_env("LINEAR_AGENT_PEER_APP_USER_IDS")
        )
        self._max_agent_handoffs = _intish(
            extra.get("max_agent_handoffs") or _str_env("LINEAR_AGENT_MAX_AGENT_HANDOFFS"),
            DEFAULT_MAX_AGENT_HANDOFFS,
        )
        self._allow_all_users = _bool_opt(extra, "allow_all_users", "LINEAR_AGENT_ALLOW_ALL_USERS", False)
        self._ack_on_created = _boolish(extra.get("ack_on_created"), True)
        self._ack_body = str(extra.get("ack_body") or DEFAULT_ACK_BODY)
        # Default ON (opt-out): moving human-delegated work into a started
        # state is Linear's published best practice.
        self._auto_start_on_delegation = _bool_opt(
            extra,
            "auto_start_on_delegation",
            "LINEAR_AGENT_AUTO_START_ON_DELEGATION",
            True,
        )
        # Default OFF (opt-in): the agent claiming unclaimed issues for itself
        # proved too eager in live testing — humans decide what the agent owns
        # unless the operator explicitly enables this.
        self._auto_self_delegate = _bool_opt(extra, "auto_self_delegate", "LINEAR_AGENT_AUTO_SELF_DELEGATE", False)
        # Default OFF: delegation already arrives as a `created` agent
        # session, so dispatching issue-update webhooks as turns too would
        # double-process every delegation. Opt in to react to issue edits.
        self._dispatch_issue_updates = _bool_opt(
            extra,
            "dispatch_issue_updates",
            "LINEAR_AGENT_DISPATCH_ISSUE_UPDATES",
            False,
        )
        # WORKAROUND (Linear platform limitation, not a permanent feature):
        # the Agent Sessions API offers no way to make a session response
        # render inline in the thread the agent was mentioned from — mid-thread
        # mentions re-anchor the session to a copied root, and agent comments
        # inside a session-hosted thread are not rendered by Linear. Only the
        # first-party @Linear assistant replies inline. When on, this posts the
        # final findings as a reply on the mention's source thread root (a
        # normal thread that does render). Linear's sourceCommentId identifies
        # the child mention, but commentCreate only accepts the root as
        # parentId, so the adapter resolves and retains that root before
        # dispatch. Default off keeps behavior identical to other third-party
        # agents (e.g. Cursor). Requires create_comments.
        # UNWIND: if Linear routes a `response` into its sourceComment thread
        # (or otherwise lets third-party agents reply inline), delete this flag,
        # its yaml/env plumbing, source-route state, and mirror send below.
        # AgentSession.sourceComment already exists, so this is plausibly
        # server-side work on Linear's part.
        self._reply_in_source_thread = _bool_opt(
            extra,
            "reply_in_source_thread",
            "LINEAR_AGENT_REPLY_IN_SOURCE_THREAD",
            False,
        )
        self._mutation_policy = _merge_policy(extra.get("mutation_policy"))
        self._auto_skills = _list_value(extra.get("auto_skills"))
        self._max_body_bytes = int(extra.get("max_body_bytes") or 1_048_576)
        self._oauth_manager = _build_oauth_manager(extra, self._access_token)
        self._client = client or _build_client(extra, self._access_token, self._oauth_manager)
        self._runner = None
        self.gateway_runner = None
        self._dedup = MessageDeduplicator(ttl_seconds=_DEDUP_TTL_SECONDS)
        # Last time an ephemeral "working" thought was posted per session —
        # rate-limits the typing indicator (see send_typing).
        self._typing_marks: dict[str, float] = {}
        # Agent Session ID -> (issue ID, source thread root comment ID). Routes
        # are consumed exactly once by send(); bounded against abandoned runs.
        self._source_thread_routes: dict[str, tuple[str, str]] = {}
        # Issue ID -> NEWEST Linear Agent Session ID seen on it. The Hermes
        # session (and every outbound chat_id) is keyed per ISSUE while Linear
        # mints a fresh agent session per mention, so replies are resolved
        # through this map to land in the current session.
        self._latest_agent_session: dict[str, str] = {}

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.warning("[linear_agent] aiohttp is not installed")
            return False
        client_configured = getattr(self._client, "configured", bool(self._access_token or self._oauth_manager))
        if not client_configured:
            logger.warning(
                "[linear_agent] Configure LINEAR_AGENT_ACCESS_TOKEN or LINEAR_AGENT_CLIENT_ID/SECRET client-credentials"
            )
            return False

        # Prevent two profiles from connecting with the same Linear credential
        # (AGENTS.md scoped-lock pitfall). Prefer the stable OAuth client_id;
        # fall back to a hash of the access token so the raw secret is never
        # written to the lock file. When neither is available there is no
        # stable identity to key on, so locking is skipped.
        lock_identity = self._scoped_lock_identity()
        if lock_identity and not self._acquire_platform_lock("linear_agent", lock_identity, "Linear Agent credential"):
            return False

        # Discover our own identity BEFORE the webhook server exists, so the
        # first delivery already has the self-echo filter and delegation
        # verification available.
        await self._discover_app_user_id()

        # client_max_size: aiohttp's own body cap (default 1 MiB) must track
        # the configured limit, or larger configured payloads 500 inside
        # request.read() instead of 413ing at the explicit check.
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._webhook_path, self._handle_webhook)
        app.router.add_post(f"/p/{{profile}}{self._webhook_path}", self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        bind_pooled_loop = getattr(self._client, "bind_pooled_loop", None)
        if bind_pooled_loop is not None:
            bind_pooled_loop()
        self._mark_connected()
        logger.info(
            "[linear_agent] Listening on %s:%d%s",
            self._host,
            self._port,
            self._webhook_path,
        )
        set_active_adapter(self)
        return True

    async def _discover_app_user_id(self) -> None:
        """Fill a missing app_user_id from Linear's viewer at connect.

        Without it the adapter can't filter its own webhook echoes or verify
        delegation. Explicit config wins; failure never blocks connect (the
        webhook secret still gates inbound).
        """
        if self._app_user_id:
            return
        try:
            data = await asyncio.wait_for(self._client.execute("query { viewer { id } }"), 10)
            self._app_user_id = str(((data or {}).get("viewer") or {}).get("id") or "").strip()
        except Exception:  # noqa: BLE001 - a Linear outage must not stop connect
            logger.warning(
                "[linear_agent] Could not auto-discover the app user id "
                "(viewer query failed); set LINEAR_AGENT_APP_USER_ID to enable "
                "self-echo filtering and delegation verification.",
                exc_info=True,
            )
            return
        if self._app_user_id:
            logger.info(
                "[linear_agent] Discovered app user id %s from Linear viewer",
                self._app_user_id,
            )

    def _scoped_lock_identity(self) -> str:
        """Stable per-credential identity for the connect scoped lock.

        Never returns the raw access token — a token-derived identity is a
        truncated SHA-256 hash so the secret never lands in the lock file.
        """
        extra = self.config.extra or {}
        client_id = _first_secret(extra, "client_id", "LINEAR_AGENT_CLIENT_ID")
        if client_id:
            return f"client:{client_id}"
        if self._access_token:
            import hashlib

            return "token:" + hashlib.sha256(self._access_token.encode("utf-8")).hexdigest()[:16]
        return ""

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - closing pooled connections is best-effort
                logger.debug("[linear_agent] Failed to close GraphQL session", exc_info=True)
        self._release_platform_lock()
        self._mark_disconnected()
        set_active_adapter(None)
        logger.info("[linear_agent] Disconnected")

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": chat_id, "type": "linear_agent_session"}

    @staticmethod
    def _issue_reply_target(agent_session_id: str) -> str | None:
        """Return the issue id when the session is a synthetic issue-update one.

        Issue-data webhooks have no real agent session (extract_context keys
        them as ``update:<issue_id>``), so replies cannot go through
        agentActivityCreate — they post as comments on the issue instead.
        """
        if agent_session_id.startswith("update:"):
            return agent_session_id[len("update:") :] or None
        return None

    def _remember_agent_session(self, context: LinearWebhookContext) -> None:
        """Track the NEWEST Linear agent session per issue, before dispatch.

        Linear mints a fresh agent session for every mention on an issue, while
        the Hermes session (and every outbound ``chat_id``) is keyed per ISSUE
        (see ``_build_source``). Recording the session on every accepted
        created/prompted webhook keeps replies in the current session instead
        of the first one that ever touched the issue.
        """
        if context.action not in ("created", "prompted"):
            return
        agent_session_id = str(context.agent_session_id or "").strip()
        issue_id = str(context.issue_id or "").strip()
        if not agent_session_id or not issue_id or self._issue_reply_target(agent_session_id):
            return
        if len(self._latest_agent_session) >= 512 and issue_id not in self._latest_agent_session:
            self._latest_agent_session.clear()  # unbounded-growth guard
        self._latest_agent_session[issue_id] = agent_session_id

    def _reply_target(self, chat_id: str, metadata: dict[str, Any] | None = None) -> tuple[str, str]:
        """Resolve an outbound ``chat_id`` (the ISSUE id) to a reply target.

        Returns ``(agent_session_id, issue_id)`` with at most one non-empty:
        the newest Linear agent session seen on that issue (→ Agent Activity),
        otherwise the issue id itself (→ issue comment). The comment fallback
        covers turns with no Linear session at all (synthetic issue-update
        sessions, cron/``LINEAR_AGENT_HOME_TARGET`` delivery) and a restarted
        gateway whose in-memory per-issue map is empty — replying somewhere
        beats failing the turn.
        """
        explicit = str((metadata or {}).get("linear_agent_session_id") or "").strip()
        if explicit:
            synthetic_issue = self._issue_reply_target(explicit)
            return ("", synthetic_issue) if synthetic_issue else (explicit, "")
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return "", ""
        synthetic_issue = self._issue_reply_target(chat_id)
        if synthetic_issue:
            return "", synthetic_issue
        agent_session_id = self._latest_agent_session.get(chat_id)
        if agent_session_id:
            return agent_session_id, chat_id
        if chat_id in self._latest_agent_session.values():
            # A bare agent-session id (direct caller, restored state).
            return chat_id, ""
        return "", chat_id

    async def _recover_reply_session(self, issue_id: str) -> str:
        """Find this app's newest Linear agent session on an issue (P13).

        The per-issue map is memory-only, so a gateway restart loses it and
        replies fall back to an issue comment. Linear still knows the
        sessions: pick the newest one whose ``appUser`` is THIS app. Other
        apps' sessions on the same issue are ignored — replying into another
        agent's session would speak for the wrong agent. Failures return ""
        so the caller keeps the existing comment fallback.
        """
        if not self._app_user_id:
            return ""
        try:
            sessions = await self._client.list_issue_agent_sessions(issue_id, limit=50)
        except Exception as exc:  # noqa: BLE001 - fallback is a comment, not a failure
            logger.warning(
                "[linear_agent] Could not recover a reply session for issue %s: %s",
                issue_id,
                exc,
            )
            return ""
        mine = [
            node
            for node in sessions or []
            if str(((node.get("appUser") or {}).get("id")) or "").strip() == self._app_user_id
            and str(node.get("id") or "").strip()
        ]
        if not mine:
            return ""
        # Linear's createdAt is an ISO-8601 UTC instant, so lexical order is
        # chronological order.
        newest = max(mine, key=lambda node: str(node.get("createdAt") or ""))
        agent_session_id = str(newest.get("id") or "").strip()
        if len(self._latest_agent_session) >= 512 and issue_id not in self._latest_agent_session:
            self._latest_agent_session.clear()
        self._latest_agent_session[issue_id] = agent_session_id
        logger.info(
            "[linear_agent] Recovered agent session %s for issue %s from Linear (restart)",
            agent_session_id,
            issue_id,
        )
        return agent_session_id

    def _recoverable_reply_target(self, chat_id: str, metadata: dict[str, Any] | None = None) -> bool:
        """Whether a missing reply session may be recovered from Linear.

        Excluded: an explicit caller-supplied session (the caller decided the
        target), and the synthetic ``update:<issue_id>`` issue-update session,
        which by design has no Linear agent session to reply into.
        """
        if str((metadata or {}).get("linear_agent_session_id") or "").strip():
            return False
        return not self._issue_reply_target(str(chat_id or "").strip())

    async def _send_issue_comment(self, issue_id: str, content: str) -> SendResult:
        try:
            comment = await self._client.create_comment(
                issue_id,
                content,
                mutation_policy=self._mutation_policy,
            )
        except Exception as exc:  # noqa: BLE001 - adapter boundary returns SendResult
            logger.warning(
                "[linear_agent] Failed to post issue-update reply as comment on %s: %s",
                issue_id,
                exc,
            )
            return SendResult(success=False, error=str(exc))
        return SendResult(
            success=True,
            message_id=str(comment.get("id") or ""),
            raw_response=comment,
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send the final Hermes response to Linear.

        ``chat_id`` is the ISSUE id (the Hermes session is keyed per issue, see
        ``_build_source``). The reply becomes a ``response`` Agent Activity in
        the newest Linear agent session seen on that issue; without a known
        session (a remaining restart, cron/home delivery, synthetic
        issue-update turns) it re-derives this app's newest session on the
        issue from Linear, and only then posts as an issue comment.
        """
        agent_session_id, issue_id = self._reply_target(chat_id, metadata)
        if not agent_session_id and issue_id and self._recoverable_reply_target(chat_id, metadata):
            recovered = await self._recover_reply_session(issue_id)
            if recovered:
                agent_session_id, issue_id = recovered, ""
        if not agent_session_id:
            if not issue_id:
                return SendResult(success=False, error="Missing Linear Agent Session ID")
            logger.info(
                "[linear_agent] No live agent session for issue %s; posting the reply as an issue comment",
                issue_id,
            )
            return await self._send_issue_comment(issue_id, content)
        activity: dict[str, Any] = {}
        chunks = _split_message(content, self.MAX_MESSAGE_LENGTH)
        try:
            # Nothing upstream chunks the non-streaming path, so oversized
            # responses are split here rather than failing the whole send.
            for chunk in chunks:
                activity = await self._client.create_response(agent_session_id, chunk)
        except Exception as exc:  # noqa: BLE001 - adapter boundary returns SendResult
            logger.warning("[linear_agent] Failed to create response activity: %s", exc)
            return SendResult(success=False, error=str(exc))

        source_reply: dict[str, Any] | None = None
        source_route = self._source_thread_routes.pop(agent_session_id, None)
        if source_route:
            source_issue_id, root_comment_id = source_route
            try:
                for chunk in chunks:
                    source_reply = await self._client.create_comment(
                        source_issue_id,
                        chunk,
                        parent_id=root_comment_id,
                        mutation_policy=self._mutation_policy,
                    )
                logger.info(
                    "[linear_agent] Mirrored final response for session %s to source thread root %s",
                    agent_session_id,
                    root_comment_id,
                )
            except Exception as exc:  # noqa: BLE001 - activity already succeeded
                logger.warning(
                    "[linear_agent] Agent response succeeded but source-thread mirror failed for %s: %s",
                    agent_session_id,
                    exc,
                )
                return SendResult(
                    success=False,
                    message_id=str(activity.get("id") or ""),
                    error=f"Agent response created, but source-thread mirror failed: {exc}",
                    raw_response=activity,
                )
        return SendResult(
            success=True,
            message_id=str(activity.get("id") or ""),
            raw_response={"agent_activity": activity, "source_comment": source_reply} if source_reply else activity,
        )

    async def send_error_activity(self, chat_id: str, body: str) -> SendResult:
        """Report a failed turn as an ``error`` Agent Activity.

        ``chat_id`` is resolved like every other outbound path (issue id →
        newest agent session id, comment fallback when there is none).
        """
        agent_session_id, issue_id = self._reply_target(chat_id)
        if not agent_session_id:
            if not issue_id:
                return SendResult(success=False, error="Missing Linear Agent Session ID")
            return await self._send_issue_comment(issue_id, body)
        try:
            activity = await self._client.create_error(agent_session_id, body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[linear_agent] Failed to create error activity: %s", exc)
            return SendResult(success=False, error=str(exc))
        return SendResult(
            success=True,
            message_id=str(activity.get("id") or ""),
            raw_response=activity,
        )

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: list | None,
        clarify_id: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Ask a clarifying question as a Linear ``elicitation`` activity.

        Elicitation moves the session to ``awaitingInput`` in Linear's UI, so
        the user sees the agent is blocked on them (per Linear's Agent
        Interaction Guidelines). The user's reply arrives as a normal
        ``prompted`` webhook and the gateway text-intercept resolves the
        clarify, exactly like the base text-fallback path.
        """
        agent_session_id, _issue_id = self._reply_target(chat_id, metadata)
        if not agent_session_id:
            # No Linear session to elicit in (synthetic issue-update turns, or
            # unknown after a restart): fall back to the base text rendering,
            # which the base send path posts as an issue comment.
            return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata=metadata)

        if choices:
            lines = [question, ""]
            for i, choice in enumerate(choices, start=1):
                lines.append(f"{i}. {choice}")
            lines.append("")
            lines.append("Reply with the number, the option text, or your own answer.")
            body = "\n".join(lines)
        else:
            body = question

        try:
            # `select` is the agent→human signal for choice elicitations
            # (Linear agent-signals contract).
            activity = await self._client.create_response(
                agent_session_id,
                body,
                response_type="elicitation",
                signal="select" if choices else None,
            )
        except Exception as exc:  # noqa: BLE001 - adapter boundary returns SendResult
            logger.warning("[linear_agent] Failed to create elicitation activity: %s", exc)
            return SendResult(success=False, error=str(exc))
        if choices:
            # Arm the text-intercept only AFTER the question is actually
            # posted — armed-but-undelivered would swallow the user's next
            # message as an answer to a question they never saw.
            from tools.clarify_gateway import mark_awaiting_text

            mark_awaiting_text(clarify_id)
        return SendResult(
            success=True,
            message_id=str(activity.get("id") or ""),
            raw_response=activity,
        )

    # Minimum seconds between ephemeral "working" thoughts per session.
    TYPING_THOUGHT_INTERVAL = 30.0

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show a working state in Linear during long turns.

        Linear has no native typing indicator; an ephemeral ``thought``
        renders as a transient status line and is replaced by the next
        activity — Linear's AIG asks for exactly this kind of immediate,
        unobtrusive feedback. Rate-limited so the typing heartbeat doesn't
        spam the session timeline. ``chat_id`` is the issue id: it resolves to
        the newest agent session on that issue, and is a no-op when there is
        none (nothing to show a working state in).
        """
        session_id, issue_id = self._reply_target(chat_id, metadata)
        if not session_id:
            return
        # Rate-limit per ISSUE (the session granularity): a follow-up agent
        # session on the same issue must not reset the indicator, and stop
        # clears it with the same chat_id the gateway uses.
        mark_key = issue_id or session_id
        now = time.time()
        if now - self._typing_marks.get(mark_key, 0.0) < self.TYPING_THOUGHT_INTERVAL:
            return
        if len(self._typing_marks) > 256:  # unbounded-growth guard
            self._typing_marks.clear()
        self._typing_marks[mark_key] = now
        try:
            await self._client.create_thought(session_id, "Working on it…", ephemeral=True)
        except Exception:  # noqa: BLE001 - typing is best-effort
            logger.debug("[linear_agent] Working-state thought failed", exc_info=True)

    def _own_profile_name(self) -> str:
        """This adapter's own Hermes profile name ("" when the gateway never named one)."""
        for attr in ("_owner_profile", "_hermes_profile_name"):
            value = getattr(self, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _accepts_profile_path(self, profile: str | None) -> bool:
        """True when a ``/p/<profile>/hermes/linear-agent`` prefix names THIS adapter.

        The prefix exists so a shared multiplexed listener can reach a named
        profile's app. On a single-profile listener a different profile names a
        bot this process does not serve — serving it here would run another
        profile's traffic under this profile's credentials and allowlists — so
        it fails closed.
        """
        name = str(profile or "").strip()
        if not name:
            return True
        own = self._own_profile_name()
        if own:
            return name == own
        try:
            from hermes_cli.profiles import profile_matches_home

            return bool(profile_matches_home(name))
        except Exception:  # noqa: BLE001 - unimportable profile helpers must fail closed
            logger.warning(
                "[linear_agent] Cannot verify URL profile %r (profile helpers unavailable); rejecting",
                name,
                exc_info=True,
            )
            return False

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "platform": "linear_agent"})

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)

        raw_body = await request.read()
        profile = request.match_info.get("profile") or None
        try:
            body, status = await self.handle_webhook(
                dict(request.headers),
                raw_body,
                profile=profile,
            )
        except Exception:
            logger.exception("[linear_agent] Unhandled webhook failure")
            return web.json_response({"error": "Webhook handling failed"}, status=500)
        return web.json_response(body, status=status)

    async def handle_webhook(
        self,
        headers: dict[str, str],
        raw_body: bytes,
        *,
        profile: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        # A /p/<profile>/ prefix must name THIS adapter's profile before the
        # body is authenticated or anything else happens (see
        # _accepts_profile_path): an unknown/foreign profile is a 404.
        if not self._accepts_profile_path(profile):
            logger.warning(
                "[linear_agent] Rejecting webhook for URL profile %r: this adapter serves %r",
                profile,
                self._own_profile_name() or "the active profile",
            )
            return {"error": "Unknown profile"}, 404

        if self._webhook_secret:
            if not verify_linear_signature(headers, raw_body, self._webhook_secret):
                logger.warning(
                    "[linear_agent] Invalid webhook signature; signature_headers=%s body_bytes=%d",
                    describe_signature_headers(headers),
                    len(raw_body),
                )
                return {"error": "Invalid signature"}, 401
        elif not self._allow_unsigned_webhooks:
            # Fail closed: without a shared secret anyone who can reach this
            # port can inject webhooks. Require explicit opt-in to run unsigned.
            logger.warning(
                "[linear_agent] Rejecting unsigned webhook: LINEAR_AGENT_WEBHOOK_SECRET "
                "is not configured. Set the secret, or set allow_unsigned_webhooks: true "
                "to accept unsigned deliveries."
            )
            return {"error": "Webhook secret not configured"}, 401

        try:
            payload = parse_json_body(raw_body)
        except LinearWebhookError as exc:
            return {"error": str(exc)}, 400

        if is_stale_body_timestamp(payload, headers):
            logger.warning("[linear_agent] Rejecting stale webhook delivery (replay guard)")
            return {"error": "Stale webhook delivery"}, 400

        # Permission-change awareness. These event
        # shapes are not publicly documented, so match loosely over type/action
        # and surface them LOUDLY instead of silently ignoring — a revoked
        # token or lost team access otherwise looks like an inexplicable outage.
        perm_event = self._permission_change_event(payload)
        if perm_event:
            return {"status": "ignored", "reason": perm_event}, 200

        # Filter unsupported deliveries BEFORE strict context extraction: with
        # extra webhook categories enabled Linear sends data-change events
        # (Comment/Project/... with action create/update/remove) that carry no
        # agentSession — those must be acknowledged (200), not 400'd, because
        # Linear retries and eventually disables webhooks on non-200 replies.
        action = normalize_action(payload)
        if action not in SUPPORTED_ACTIONS:
            return {"status": "ignored", "action": action or "unknown"}, 200
        if action == "update" and payload.get("type") != "Issue":
            return {
                "status": "ignored",
                "reason": f"unhandled update type {payload.get('type')!r}",
            }, 200

        try:
            context = extract_context(payload, headers)
        except LinearWebhookError as exc:
            return {"error": str(exc)}, 400

        # Workspace restriction: one Linear app (and webhook secret) can be
        # installed in multiple workspaces, so a valid signature does not
        # imply the configured workspace. Enforced before authorization and
        # all side effects, and fail closed: a delivery that cannot name the
        # configured workspace is treated as foreign. 200 so Linear doesn't
        # retry legitimate traffic from the app's other workspaces.
        if self._workspace_id and context.workspace_id != self._workspace_id:
            if not context.workspace_id:
                logger.warning(
                    "[linear_agent] Ignoring webhook with no workspace id (configured workspace: %s)",
                    self._workspace_id,
                )
                return {"status": "ignored", "reason": "missing workspace"}, 200
            logger.warning(
                "[linear_agent] Ignoring webhook from workspace %s (configured workspace: %s)",
                context.workspace_id,
                self._workspace_id,
            )
            return {"status": "ignored", "reason": "other workspace"}, 200

        if context.action == "update":
            # Issue-data webhooks fire for EVERY update, including the ones
            # this agent just made. Without a self-actor filter each of the
            # agent's own mutations spawns a fresh session reviewing its own
            # change — a runaway feedback loop.
            if self._app_user_id and context.actor_user_id == self._app_user_id:
                return {"status": "ignored", "reason": "self-update"}, 200
            if not self._app_user_id:
                logger.warning(
                    "[linear_agent] Processing issue-update webhook without "
                    "LINEAR_AGENT_APP_USER_ID set — cannot filter the agent's "
                    "own updates; set it to avoid self-triggered sessions."
                )
            if not context.issue_id:
                return {"status": "ignored", "reason": "update without issue id"}, 200

        # Sender fallback for agent-to-agent handoffs: such sessions name no
        # creator and no activity author, so resolve the session comment's
        # author BEFORE authorization (fail closed if it cannot be resolved).
        context = await self._resolve_sender(context)

        if not self._webhook_sender_authorized(context):
            logger.warning(
                "[linear_agent] Rejected webhook for unauthorized user/team user=%s team=%s",
                context.actor_user_id,
                context.team_id,
            )
            return {"error": "Unauthorized Linear user or team"}, 403

        if self._dedup.is_duplicate(context.delivery_id):
            return {
                "status": "duplicate",
                "delivery_id": context.delivery_id,
            }, 200

        # Peer handoff cap: a `created` session from another agent app is only
        # dispatched while the issue still has few app-initiated sessions.
        # Humans never hit this path, and the count is read from Linear so a
        # restart cannot reset it.
        if context.action == "created" and self._is_peer_sender(context):
            refusal = await self._enforce_handoff_cap(context)
            if refusal is not None:
                return {**refusal, "delivery_id": context.delivery_id}, 200

        # Record the newest agent session on this issue BEFORE any reply path
        # can run (dispatch, stop confirmation, ack): the Hermes session and
        # every outbound chat_id are per ISSUE, so replies resolve through it.
        self._remember_agent_session(context)

        # Human→agent "stop" signal: halt immediately, make no further writes
        # beyond the required confirmation, and do NOT dispatch the activity
        # body as a prompt (Linear agent-signals contract).
        if context.action == "prompted" and context.signal == "stop":
            return await self._handle_stop_signal(context, profile=profile)

        # Linear requires the webhook response within 5 seconds and an
        # acknowledging activity within 10 (else the session is marked
        # unresponsive). Run the ack concurrently with dispatch instead of serially
        # so a slow Linear API can't stack the two on the response path.
        ack_task: asyncio.Task | None = None
        if context.action == "created" and self._ack_on_created:
            ack_task = asyncio.create_task(self._send_ack_thought(context))

        # Auto-start delegated issues concurrently with dispatch (created +
        # update actions only; the guard inside verifies delegation). Runs
        # after the self-actor filter and dedup so we never start off our own
        # echo or a replay.
        auto_start_task: asyncio.Task | None = None
        if (
            context.action in ("created", "update")
            and self._auto_start_on_delegation
            and self._mutation_policy.get("update_issues")
        ):
            auto_start_task = asyncio.create_task(self._maybe_auto_start_issue(context))

        # Issue-data webhooks are a generic change firehose, and delegation
        # already arrives as a real `created` agent session — so by default an
        # update webhook only feeds auto-start, never a full agent turn
        # (dispatching both would run two concurrent turns per delegation).
        # dispatch_issue_updates: true opts in to reacting to issue edits via
        # a synthetic comment-reply session.
        if context.action == "update" and not self._dispatch_issue_updates:
            await self._await_side_tasks(auto_start_task)
            return {
                "status": "accepted",
                "action": context.action,
                "delivery_id": context.delivery_id,
                "reason": "auto-start only",
            }, 200

        if self._reply_in_source_thread and context.source_comment_id and context.issue_id:
            await self._prepare_source_thread_route(context)

        message = (
            build_created_prompt(context, payload)
            if context.action == "created"
            else build_prompted_message(context)
            if context.action == "prompted"
            else build_update_prompt(context)
        )

        try:
            await self._dispatch_linear_message(context, message, payload, profile=profile)
        except Exception as exc:  # noqa: BLE001 - send an Agent Activity error when possible
            logger.exception("[linear_agent] Dispatch failed for session %s", context.agent_session_id)
            await self.send_error_activity(
                context.issue_id or context.agent_session_id,
                f"The assistant failed to process this Linear Agent Session: {exc}",
            )
            # 200: the delivery itself was handled (Linear retries non-200,
            # and repeated failures disable the webhook).
            return {"status": "error", "delivery_id": context.delivery_id}, 200
        finally:
            # Both side tasks swallow their own exceptions. Wait only briefly:
            # Linear expects the webhook HTTP response within 5 seconds, and
            # these tasks can involve Linear round-trips (30s client timeout
            # plus a possible 429 retry). Anything slower detaches and
            # finishes in the background.
            await self._await_side_tasks(ack_task, auto_start_task)

        # 200 (not 202): Linear counts any non-200 reply as a failed delivery,
        # retries it 3x, and can disable a persistently "failing" webhook.
        return {
            "status": "accepted",
            "action": context.action,
            "delivery_id": context.delivery_id,
            "agent_session_id": context.agent_session_id,
        }, 200

    async def _prepare_source_thread_route(self, context: LinearWebhookContext) -> None:
        """Resolve and retain the renderable root for one deterministic mirror."""
        if not self._mutation_policy.get("create_comments"):
            logger.warning(
                "[linear_agent] reply_in_source_thread is enabled but mutation_policy.create_comments is false"
            )
            return
        try:
            root_id = await self._client.get_comment_thread_root_id(context.source_comment_id)
        except Exception as exc:  # noqa: BLE001 - dispatch should still proceed
            logger.warning(
                "[linear_agent] Could not resolve source thread root for session %s: %s",
                context.agent_session_id,
                exc,
            )
            return
        if not root_id:
            logger.warning(
                "[linear_agent] Source comment %s was not found for session %s",
                context.source_comment_id,
                context.agent_session_id,
            )
            return
        if len(self._source_thread_routes) >= 256:
            self._source_thread_routes.clear()
        self._source_thread_routes[context.agent_session_id] = (context.issue_id, root_id)

    async def _resolve_sender(self, context: LinearWebhookContext) -> LinearWebhookContext:
        """Resolve the sender for a session that does not name a usable one.

        Resolution order (P12 for handoffs, P14 for automation-rule
        delegation): ``agentActivity.userId`` (unless it is this app acting as
        Linear's machine prompt on a ``created`` session), then
        ``agentSession.creatorId``, then the session comment's author
        (payload field, else a Linear lookup), then the issue's creator
        (payload field, else a Linear lookup).

        Linear leaves the session creator null both for machine handoffs
        (another agent app is the real sender, readable from the session's
        comment) and for sessions started by an automation rule or delegation
        (the issue's creator is the human the work is for). Failing to resolve
        anything — including a failed lookup — leaves the sender empty, which
        the fail-closed authorization then rejects.
        """
        if context.action not in ("created", "prompted") or self._issue_reply_target(context.agent_session_id):
            return context

        sender = str(context.actor_user_id or "").strip()
        source = context.sender_source or ("agentActivity.userId" if sender else "")
        if (
            sender
            and context.action == "created"
            and source == "agentActivity.userId"
            and self._app_user_id
            and sender == self._app_user_id
        ):
            # A Linear automation rule / delegation starts the session for
            # this app and Linear authors the opening prompt as our own app
            # user: that activity is the machine prompt, not a sender, so keep
            # resolving. On any other action the same value is our own activity
            # echoing back and must stay a rejected sender.
            logger.info(
                "[linear_agent] Session %s was prompted by this app itself "
                "(automation rule or delegation); not treating the activity author as a sender",
                context.agent_session_id,
            )
            sender, source = "", ""

        if not sender:
            sender, source = str(context.session_creator_id or "").strip(), "agentSession.creatorId"
        if not sender:
            sender = str(context.comment_user_id or "").strip()
            if not sender:
                try:
                    sender = str(
                        await self._client.get_agent_session_comment_user_id(context.agent_session_id) or ""
                    ).strip()
                except Exception as exc:  # noqa: BLE001 - unresolved sender must stay unauthorized
                    logger.warning(
                        "[linear_agent] Could not resolve the comment author for session %s: %s",
                        context.agent_session_id,
                        exc,
                    )
                    sender = ""
            if sender:
                source = "agentSession.comment.userId"
        if not sender:
            sender, source = await self._issue_creator_sender(context)

        if not sender:
            logger.info(
                "[linear_agent] Session %s names no sender (activity, creator, comment and issue creator all unresolved); refusing",
                context.agent_session_id,
            )
            return context
        logger.info(
            "[linear_agent] Session %s sender resolved to %s from %s",
            context.agent_session_id,
            sender,
            source or "unknown",
        )
        return replace(context, actor_user_id=sender, sender_source=source)

    async def _issue_creator_sender(self, context: LinearWebhookContext) -> tuple[str, str]:
        """Sender from the issue's creator (P14), or ``("", "")``.

        The agent-session payload serializes the issue without its creator, so
        this normally costs one GraphQL lookup. A creator that is this app
        itself is not a sender: leave it empty so authorization fails closed
        instead of accepting the app's own id.
        """
        issue_id = str(context.issue_id or "").strip()
        if not issue_id:
            return "", ""
        creator_id = str(context.issue_creator_id or "").strip()
        if not creator_id:
            try:
                creator_id = str(await self._client.get_issue_creator_user_id(issue_id) or "").strip()
            except Exception as exc:  # noqa: BLE001 - unresolved sender must stay unauthorized
                logger.warning(
                    "[linear_agent] Could not resolve the creator of issue %s for session %s: %s",
                    issue_id,
                    context.agent_session_id,
                    exc,
                )
                return "", ""
        if not creator_id or creator_id == self._app_user_id:
            return "", ""
        return creator_id, "issue.creatorId"

    def _is_peer_sender(self, context: LinearWebhookContext) -> bool:
        """True when the sender is a listed peer app (and not ourselves)."""
        sender = str(context.actor_user_id or "").strip()
        if not sender or sender == self._app_user_id:
            return False
        return sender in self._peer_app_user_ids

    @staticmethod
    def _session_initiated_by_app(node: dict[str, Any]) -> bool:
        """Whether an agent session was started by an app rather than a human.

        ``creator`` is null for app-created sessions; the fallback reads the
        session comment's author, which is the other agent app in a handoff.
        """
        creator = node.get("creator") or {}
        if creator.get("app") is True:
            return True
        if creator:
            return False
        return ((node.get("comment") or {}).get("user") or {}).get("app") is True

    async def _enforce_handoff_cap(self, context: LinearWebhookContext) -> dict[str, Any] | None:
        """Refuse a peer session once the issue is saturated with handoffs.

        Stateless by design: the count comes from Linear's own session history
        for the issue, so a restarted gateway cannot lose the cap. Returns an
        accepted-webhook body when the session was refused, else None.
        """
        issue_id = str(context.issue_id or "").strip()
        if not issue_id:
            return None
        try:
            sessions = await self._client.list_issue_agent_sessions(issue_id, limit=50)
        except Exception as exc:  # noqa: BLE001 - fail CLOSED for peer sessions
            logger.warning(
                "[linear_agent] Refusing peer session %s: handoff count unavailable (%s)",
                context.agent_session_id,
                exc,
            )
            await self._post_session_activity(
                context.agent_session_id,
                "Could not verify this issue's agent-to-agent handoff count, "
                "so this handoff was refused. Tom needs to step in.",
                error=True,
            )
            return {"status": "refused", "reason": "handoff count unavailable"}
        count = sum(1 for node in sessions if self._session_initiated_by_app(node))
        if count <= self._max_agent_handoffs:
            return None
        logger.warning(
            "[linear_agent] Handoff limit reached on issue %s: %s agent-to-agent handoffs (max %s); refusing peer session %s",
            issue_id,
            count,
            self._max_agent_handoffs,
            context.agent_session_id,
        )
        await self._post_session_activity(
            context.agent_session_id,
            f"Handoff limit reached: this issue already has {count} agent-to-agent handoffs. "
            "Tom needs to step in.",
        )
        return {"status": "refused", "reason": "handoff limit reached", "handoffs": count}

    async def _post_session_activity(self, agent_session_id: str, body: str, *, error: bool = False) -> None:
        try:
            if error:
                await self._client.create_error(agent_session_id, body)
            else:
                await self._client.create_response(agent_session_id, body)
        except Exception as exc:  # noqa: BLE001 - the refusal itself must not 500
            logger.warning(
                "[linear_agent] Failed to post handoff refusal into session %s: %s",
                agent_session_id,
                exc,
            )

    def _webhook_sender_authorized(self, context: LinearWebhookContext) -> bool:
        """Authorize the sender BEFORE any webhook side effect (fail closed).

        Primary: the gateway-registered authorization callback — the exact
        chain the dispatch layer uses (platform env allowlists including the
        ``*`` wildcard, ``GATEWAY_ALLOWED_USERS``, pairing-store grants) — so
        the grants the docs promise stay reachable and the two layers cannot
        drift. Adapter-local YAML ``allowed_users``/``allow_all_users`` still
        grant as a union (the gateway can't see YAML extras), and
        ``allowed_teams`` always narrows, even over a gateway grant. When no
        callback is registered (standalone runs, unit tests), the local
        fail-closed check decides alone.
        """
        sender = str(context.actor_user_id or "").strip()
        # An agent-session event whose sender could not be resolved (no
        # activity author we may trust, no creator, no comment author, no issue
        # creator, or a failed lookup) fails closed even under
        # ``allow_all_users``: "all users" grants humans, not an anonymous
        # session (P14).
        if not sender and context.action in ("created", "prompted"):
            logger.warning(
                "[linear_agent] Rejecting webhook for session %s: no sender could be resolved",
                context.agent_session_id,
            )
            return False
        # Our own app user is never a valid sender: a session the agent
        # created for itself (or a handoff it addressed to itself) must not
        # re-enter as new work, whatever the allowlists say.
        if sender and sender == self._app_user_id:
            logger.warning(
                "[linear_agent] Rejecting webhook whose sender is this app itself (%s)",
                sender,
            )
            return False
        if is_authorized(
            context,
            allowed_users=self._allowed_users,
            peer_app_user_ids=self._peer_app_user_ids,
            allowed_teams=self._allowed_teams,
            allow_all_users=self._allow_all_users,
        ):
            return True
        gateway_verdict = self._is_sender_authorized(context.actor_user_id, "thread", context.agent_session_id)
        # allow_all_users=True expresses "the user-grant is satisfied by the
        # gateway chain"; is_authorized then applies only the team narrowing.
        return gateway_verdict is True and is_authorized(
            context,
            allowed_users=self._allowed_users,
            peer_app_user_ids=self._peer_app_user_ids,
            allowed_teams=self._allowed_teams,
            allow_all_users=True,
        )

    # How long the webhook response waits on best-effort side tasks (ack
    # thought, auto-start) before detaching them to the background. Kept
    # well under Linear's 5-second webhook deadline.
    SIDE_TASK_WAIT_SECONDS = 2.0

    async def _await_side_tasks(self, *tasks: asyncio.Task | None) -> None:
        """Await best-effort side tasks under ONE shared budget; detach the slow.

        The budget is shared (a single ``asyncio.wait``) rather than paid per
        task, so even with every side task slow the webhook response is
        delayed by at most SIDE_TASK_WAIT_SECONDS total. ``asyncio.wait``
        never cancels, so detached tasks keep running in the background with
        a strong reference held until done.
        """
        pending = {task for task in tasks if task is not None}
        if not pending:
            return
        done, slow = await asyncio.wait(pending, timeout=self.SIDE_TASK_WAIT_SECONDS)
        for task in slow:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        for task in done:
            # Side tasks swallow their own errors; retrieve any stray
            # exception so asyncio never logs "exception was never retrieved".
            if not task.cancelled() and task.exception():
                logger.debug("[linear_agent] side task failed", exc_info=task.exception())

    async def _send_ack_thought(self, context: LinearWebhookContext) -> None:
        try:
            # Ephemeral so the ack disappears once the real response arrives.
            await self._client.create_thought(
                context.agent_session_id,
                self._ack_body,
                ephemeral=True,
            )
        except Exception as exc:  # noqa: BLE001 - ack is best-effort
            logger.warning("[linear_agent] Thought acknowledgement failed: %s", exc)

    @staticmethod
    def _permission_change_event(payload: dict[str, Any]) -> str | None:
        """Classify OAuth-revoked / team-access-changed webhook events.

        Returns a short reason string (and logs a WARNING) when the payload
        looks like a permission change, else None. Matching is intentionally
        loose because Linear does not publish these payload shapes.
        """
        haystack = " ".join(str(payload.get(key) or "") for key in ("type", "action", "event")).lower()
        if "revoked" in haystack:
            logger.warning(
                "[linear_agent] OAuth access revoked event received — re-authorize the Linear app (see plugin README)"
            )
            return "revoked"
        if "teamaccesschanged" in haystack:
            logger.warning(
                "[linear_agent] Team access changed for the Linear app; verify allowed_teams/workspace configuration"
            )
            return "teamAccessChanged"
        return None

    async def _maybe_auto_start_issue(self, context: LinearWebhookContext) -> None:
        """Move a freshly delegated issue to its first `started` state.

        Linear best practice: delegated work moves into progress. Best-effort
        — never raises, never blocks the webhook response beyond one Linear
        round-trip. Guards: flag on, update_issues policy on, then delegation
        VERIFIED on a freshly fetched issue — the webhook payload is never
        positive evidence. Triage-state issues are left for humans; claiming
        (`auto_self_delegate`, default off) needs a real `created` session on
        an unclaimed issue.
        """
        try:
            if not self._auto_start_on_delegation:
                return
            if not self._mutation_policy.get("update_issues"):
                logger.debug("[linear_agent] auto-start skipped: update_issues disabled")
                return
            if not context.issue_id:
                return
            if context.action == "update":
                if not self._app_user_id:
                    # Without our own id we can never verify delegation, and
                    # claiming only applies to created sessions — the fetch
                    # below could not change the outcome.
                    logger.debug(
                        "[linear_agent] auto-start skipped: LINEAR_AGENT_APP_USER_ID unset — cannot verify delegation"
                    )
                    return
                if context.issue_delegate_known and context.issue_delegate_id != self._app_user_id:
                    # The update payload itself shows the issue delegated to
                    # someone else (or nobody) — a definitive negative, so
                    # skip the verification fetch. Every EDIT in the
                    # workspace fires one of these webhooks, so this is the
                    # hot path. Positive or unserialized delegates still
                    # verify against a fresh fetch below.
                    return
            issue = await self._client.get_issue(id=context.issue_id)
            if not issue:
                return
            state_type = ((issue.get("state") or {}).get("type") or "").lower()
            if state_type in {"started", "completed", "canceled", "triage"}:
                # INFO: "delegated but didn't move" is the question operators
                # actually ask; the answer must be in the logs.
                logger.info(
                    "[linear_agent] auto-start skipped for %s: already in a '%s' state",
                    issue.get("identifier") or context.issue_id,
                    state_type,
                )
                return
            delegate_id = str(((issue.get("delegate") or {}).get("id")) or "")
            # Delegation must be VERIFIED on the fetched issue: an "update"
            # webhook fires for ANY issue edit (labels, description, ...), so
            # the action alone is not evidence of delegation — treating it as
            # such would auto-start (and, opted in, claim) issues on
            # unrelated edits.
            delegated_to_us = bool(self._app_user_id) and delegate_id == self._app_user_id
            # Self-delegation (opt-in) applies only when the agent is actually
            # starting work: a real `created` agent session on an unclaimed
            # issue. Generic issue-update webhooks never claim.
            claiming = (
                self._auto_self_delegate and context.action == "created" and not delegate_id and bool(self._app_user_id)
            )
            if not (delegated_to_us or claiming):
                return
            team_id = (issue.get("team") or {}).get("id")
            if not team_id:
                return
            target = await self._client.first_started_state(team_id)
            if not target:
                return
            input_data: dict[str, Any] = {"stateId": target["id"]}
            if claiming:
                input_data["delegateId"] = self._app_user_id
            await self._client.update_issue(context.issue_id, input_data, mutation_policy=self._mutation_policy)
            logger.info(
                "[linear_agent] auto-started %s → %s",
                issue.get("identifier") or context.issue_id,
                target.get("name") or target["id"],
            )
        except Exception:  # noqa: BLE001 - auto-start is best-effort
            logger.warning(
                "[linear_agent] Auto-start failed for issue %s",
                context.issue_id,
                exc_info=True,
            )

    def _build_source(self, context: LinearWebhookContext, *, profile: str | None = None):
        """Build the SessionSource for a Linear context.

        ``chat_id`` is the ISSUE id, NOT the Linear agent session id. Linear
        mints a fresh agent session for every mention on an issue, so keying
        the Hermes session on it would give each follow-up a brand-new session
        — no memory of the work already done, and no way for a follow-up to
        interrupt its own predecessor. The gateway derives its session key from
        the source (``gateway.session.build_session_key``), so the issue id
        must be in ``chat_id`` for the store, the busy path and the interrupt
        path to agree.

        ``thread_id`` repeats the issue id on purpose: ``build_session_key``
        keys a THREAD as one shared session (per-user isolation only via
        ``thread_sessions_per_user``), which is what "one Hermes session per
        issue" means — every authorized user on the issue shares it unless the
        operator opts into per-user thread sessions. Replies are resolved back
        to the newest Linear agent session by ``_reply_target``.
        """
        session_id = context.thread_id or context.agent_session_id
        source = self.build_source(
            chat_id=session_id,
            chat_name=context.chat_name,
            chat_type="thread",
            user_id=context.actor_user_id or self._app_user_id,
            user_name=context.actor_user_name or "Linear user",
            thread_id=session_id or None,
            guild_id=context.workspace_id or self._workspace_id,
            message_id=context.delivery_id,
        )
        if profile:
            source.profile = profile
        return source

    def _session_key_for(self, context: LinearWebhookContext, *, profile: str | None = None) -> str:
        """Reconstruct the session key that handle_message uses for dispatch.

        Routes through the base adapter's ``_source_session_key`` (identity
        canonicalization, grouping flags and profile namespacing) exactly like
        the gateway runner's own key derivation, so the interrupt targets the
        ``_active_sessions`` entry the running turn holds.
        """
        return super()._source_session_key(self._build_source(context, profile=profile))

    async def _handle_stop_signal(
        self, context: LinearWebhookContext, *, profile: str | None = None
    ) -> tuple[dict[str, Any], int]:
        """Respect a human ``stop`` signal (Linear agent-signals contract).

        Halts any in-flight turn for this session, makes no further Linear
        writes beyond the single required confirmation activity, and returns
        200 so Linear records the stop as handled.
        """
        session_id = context.agent_session_id
        # Same source (hence same key and same chat_id) dispatch uses.
        source = self._build_source(context, profile=profile)
        # Interrupt the running turn via the base adapter's own interrupt API
        # (sets the session's interrupt Event and stops typing); no-op if no
        # turn is active, which keeps stop idempotent.
        try:
            session_key = super()._source_session_key(source)
            had_active_turn = session_key in getattr(self, "_active_sessions", {})
            logger.info(
                "[linear_agent] Stop signal for session %s (active turn: %s)",
                session_id,
                had_active_turn,
            )
            # chat_id is the ISSUE id — the same id the gateway keys typing by.
            await self.interrupt_session_activity(session_key, source.chat_id)
        except Exception:  # noqa: BLE001 - stop must still confirm even if interrupt fails
            logger.warning(
                "[linear_agent] Failed to interrupt session %s on stop signal",
                session_id,
                exc_info=True,
            )
        # Stop the typing heartbeat for this issue.
        self._typing_marks.pop(str(source.chat_id or session_id), None)
        # Emit the single confirming final activity Linear requires on stop.
        try:
            await self._client.create_response(
                session_id,
                "Stopped — I will take no further actions in this session. Ask again if you'd like me to continue.",
            )
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            logger.warning(
                "[linear_agent] Failed to post stop confirmation for %s: %s",
                session_id,
                exc,
            )
        return {
            "status": "stopped",
            "delivery_id": context.delivery_id,
            "agent_session_id": session_id,
        }, 200

    async def _dispatch_linear_message(
        self,
        context: LinearWebhookContext,
        message: str,
        payload: dict[str, Any],
        *,
        profile: str | None = None,
    ) -> None:
        source = self._build_source(context, profile=profile)
        event = MessageEvent(
            text=message,
            message_type=MessageType.TEXT,
            source=source,
            raw_message={
                "linear": context.metadata(),
                "payload": payload,
            },
            message_id=context.delivery_id,
            auto_skill=list(self._auto_skills) if self._auto_skills else None,
        )

        # The linear_agent toolset reaches sessions through the gateway's
        # platform-toolset resolution (plugin platforms default to their own
        # registered toolset), not through the event object.
        await self.handle_message(event)


# Issue IDs (UUIDs) and identifiers (e.g. "ENG-123") only — anything else is
# a misconfigured LINEAR_AGENT_HOME_TARGET, not a valid comment target.
_ISSUE_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: str | None = None,
    media_files: list | None = None,
    force_document: bool = False,
) -> dict[str, Any]:
    """Post a message as a comment on a Linear issue without a live adapter.

    Used for out-of-process cron delivery (``deliver=linear_agent``): ``chat_id``
    is the ``LINEAR_AGENT_HOME_TARGET`` value — an issue ID or identifier such
    as ``ENG-123``. Credentials resolve exactly like adapter startup (config
    token, cached OAuth token, or LINEAR_AGENT_* env).

    ``thread_id``, ``media_files``, and ``force_document`` are accepted for
    signature parity; delivery is text-only comments.
    """
    target = str(chat_id or "").strip()
    if not target:
        return {
            "error": "linear_agent standalone send: set LINEAR_AGENT_HOME_TARGET to an issue ID or identifier (e.g. ENG-123)"
        }
    if not _ISSUE_TARGET_RE.match(target):
        return {"error": f"linear_agent standalone send: invalid issue target {target!r}"}

    extra = getattr(pconfig, "extra", None) or {}
    access_token = _resolve_access_token(getattr(pconfig, "token", ""), extra)
    client = _build_client(extra, access_token, _build_oauth_manager(extra, access_token))
    if not client.configured:
        return {"error": "linear_agent standalone send: no LINEAR_AGENT credentials configured"}
    try:
        # Cron delivery targets are explicit operator configuration, so this
        # does not consult the runtime mutation_policy (which governs
        # model-initiated writes inside agent sessions).
        comment = await client.create_comment(
            target,
            message,
            mutation_policy={"create_comments": True},
        )
    except Exception as exc:  # noqa: BLE001 - standalone boundary returns a dict
        return {"error": f"linear_agent standalone send failed: {exc}"}
    return {"success": True, "message_id": str(comment.get("id") or ""), "issue": target}


def _setup_graphql(token: str, query: str, timeout: float = 15.0) -> dict[str, Any]:
    """Small synchronous GraphQL call for the interactive setup wizard.

    Setup runs in the CLI (no event loop), so this uses urllib like the
    oauth helpers do. Raises on transport/GraphQL errors.
    """
    import json as _json
    import urllib.request

    req = urllib.request.Request(
        DEFAULT_LINEAR_GRAPHQL_URL,
        data=_json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        data = _json.loads(resp.read().decode("utf-8"))
    if data.get("errors"):
        msgs = "; ".join(str(e.get("message", e)) for e in data["errors"])
        raise RuntimeError(f"Linear GraphQL errors: {msgs}")
    return data.get("data") or {}


def interactive_setup() -> None:
    """Guided Linear Agent setup for `hermes gateway setup`.

    Beyond credential entry, this verifies the credentials against Linear,
    auto-detects the app user ID (which powers the self-echo filter and
    delegation detection), and offers the workspace member list when
    building the authorization allowlist.
    """
    from hermes_cli.cli_output import (
        print_header,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.setup import prompt_choice

    print_header("Linear Agent")
    existing = get_env_value("LINEAR_AGENT_CLIENT_ID") or get_env_value("LINEAR_AGENT_ACCESS_TOKEN")
    if existing:
        print_success("Linear Agent credentials are already configured.")
        if not prompt_yes_no("Reconfigure Linear Agent?", False):
            return

    # ── Credentials ──
    method = prompt_choice(
        "Choose auth method",
        [
            "OAuth client credentials (Recommended — auto-refreshing app-actor tokens)",
            "Static app access token",
        ],
        default=0,
        description=(
            "Create the OAuth app in Linear: Settings → API → Applications. "
            "Install with actor=app and scopes app:mentionable, app:assignable, read, write."
        ),
    )

    token = ""
    if method == 0:
        client_id = prompt("Linear OAuth client ID")
        if not client_id:
            return
        client_secret = prompt("Linear OAuth client secret", password=True)
        if not client_secret:
            return
        save_env_value("LINEAR_AGENT_CLIENT_ID", client_id)
        save_env_value("LINEAR_AGENT_CLIENT_SECRET", client_secret)
        try:
            from hermes_cli.config import get_env_path

            from .oauth import issue_client_credentials_token

            result = issue_client_credentials_token(
                client_id=client_id,
                client_secret=client_secret,
                env_path=get_env_path(),
            )
            # The helper never returns token values; read the cached token
            # it persisted so the verify/auto-detect steps below can run.
            token = str(read_auth_token(Path(result["auth_path"])).get("access_token") or "")
            print_success("Verified: minted an app-actor token via client credentials.")
        except Exception as exc:  # noqa: BLE001 - setup keeps going without verify
            print_warning(f"Could not mint a token yet ({exc}). Continuing — Hermes retries at runtime.")
    else:
        token = prompt("Linear app access token", password=True)
        if not token:
            return
        save_env_value("LINEAR_AGENT_ACCESS_TOKEN", token)

    # ── Webhook secret (fail-closed) ──
    webhook_secret = prompt("Webhook signing secret (from the Linear app's webhook settings)", password=True)
    if webhook_secret:
        save_env_value("LINEAR_AGENT_WEBHOOK_SECRET", webhook_secret)
    else:
        print_warning(
            "No webhook secret: Hermes REJECTS unsigned webhooks by default. "
            "Set LINEAR_AGENT_WEBHOOK_SECRET later, or allow_unsigned_webhooks: true "
            "for throwaway local testing only."
        )

    # ── Verify + auto-detect the app user ID ──
    if token:
        try:
            viewer = _setup_graphql(token, "{ viewer { id name } }").get("viewer") or {}
            if viewer.get("id"):
                save_env_value("LINEAR_AGENT_APP_USER_ID", viewer["id"])
                print_success(
                    f'Authenticated as "{viewer.get("name") or "agent"}" — saved '
                    "LINEAR_AGENT_APP_USER_ID (filters webhook echoes of the agent's "
                    "own writes and powers delegation detection)."
                )
        except Exception as exc:  # noqa: BLE001
            print_warning(
                f"Could not auto-detect the app user ID ({exc}). Set "
                "LINEAR_AGENT_APP_USER_ID manually — without it, the agent's own "
                "writes echo back as new sessions."
            )

    # ── Authorization (gateway layer is fail-closed) ──
    auth = prompt_choice(
        "Who may drive the agent?",
        [
            "Specific Linear users (recommended)",
            "Any workspace member",
            "Decide later (everyone is denied until configured)",
        ],
        default=0,
    )
    if auth == 0:
        picked = ""
        if token:
            try:
                users = (
                    _setup_graphql(
                        token,
                        "{ users(first: 25) { nodes { id name displayName } } }",
                    )
                    .get("users", {})
                    .get("nodes", [])
                )
                if users:
                    print_success("Workspace members:")
                    for u in users:
                        print(f"    {u['id']}  {u.get('name') or u.get('displayName') or ''}")
            except Exception:  # noqa: BLE001 - listing is a convenience only
                pass
        picked = prompt("Allowed Linear user IDs (comma-separated)")
        if picked:
            save_env_value("LINEAR_AGENT_ALLOWED_USERS", picked)
    elif auth == 1:
        save_env_value("LINEAR_AGENT_ALLOW_ALL_USERS", "true")
        print_warning(
            "Any workspace member can now drive the agent — writes stay off until mutation_policy keys are enabled."
        )

    # ── Optional cron delivery target ──
    home = prompt("Issue for cron/notification delivery, e.g. ENG-123 (optional)")
    if home:
        save_env_value("LINEAR_AGENT_HOME_TARGET", home)

    print_success("Linear Agent configured!")
    print("  Next steps:")
    print("  1. Enable the 'Agent session' (and optionally 'Issues') webhook")
    print("     categories on the Linear app.")
    print("  2. Point the webhook at: https://<your-host>/hermes/linear-agent")
    print("     (local dev: cloudflared tunnel --url http://localhost:8651)")
    print("  3. Add `linear_agent: {enabled: true}` + mutation_policy keys to")
    print("     config.yaml (all writes are off until you enable them).")
    print("  4. Restart the gateway, then @-mention the agent on an issue.")


def register(ctx) -> None:
    """Plugin entry point called by Hermes plugin discovery."""
    # See tools.register_tools_with_context: tracks the toolset as
    # plugin-provided so `hermes tools` can toggle it.
    from . import tools as _tools

    _tools.register_tools_with_context(ctx)
    ctx.register_platform(
        name="linear_agent",
        label="Linear Agent",
        adapter_factory=lambda cfg: LinearAgentAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="LINEAR_AGENT_HOME_TARGET",
        # Out-of-process cron delivery: posts the result as a comment on the
        # LINEAR_AGENT_HOME_TARGET issue. Without this hook, deliver=linear_agent
        # cron jobs fail with "No live adapter" when cron runs separately.
        standalone_sender_fn=_standalone_send,
        setup_fn=interactive_setup,
        allowed_users_env="LINEAR_AGENT_ALLOWED_USERS",
        allow_all_env="LINEAR_AGENT_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="LA",
        pii_safe=True,
        platform_hint=(
            "You are responding in a Linear Agent Session. Your final response "
            "text is delivered directly into the session's Linear thread — put "
            "the ACTUAL content of your reply there, never a narration like "
            "'posted X in the thread'. Do NOT use linear_agent_create_comment "
            "to talk to the user in the CURRENT session; that duplicates the "
            "conversation and buries your reply. Comment tools are for OTHER "
            "issues, when the user explicitly asks for a comment, or when a "
            "'Reply in the source thread' instruction is present in your prompt "
            "(then follow it exactly). To ask "
            "the user something, ask directly in your response (or via your "
            "clarify tool) — the session waits for their reply. "
            "For ALL Linear reads "
            "and writes, always prefer the linear_agent_* toolset over the generic "
            "mcp_linear_* tools — linear_agent_* tools authenticate as the Linear "
            "Agent, so changes are attributed to the agent in Linear's history. "
            "Writes: linear_agent_update_issue, linear_agent_create_comment, "
            "linear_agent_create_issue, linear_agent_create_project, "
            "linear_agent_update_project, linear_agent_create_project_update, and "
            "linear_agent_save_* tools for documents, initiatives, releases, "
            "milestones, status updates, and customer needs. Reads: "
            "linear_agent_list_* / linear_agent_get_* tools for teams, issues, "
            "projects, cycles, users, labels, comments, attachments, customers, "
            "initiatives, and releases. Write tools are gated by the operator's "
            "mutation_policy and may be disabled; if a write is rejected, report "
            "the policy message rather than retrying. When changing an issue's "
            "status you may pass the state NAME (e.g. state: 'Done') — it is "
            "resolved to the correct stateId automatically; likewise pass "
            "priority as a NAME (e.g. priority: 'Low') — Linear's numeric scale "
            "is 0=None, 1=Urgent, 2=High, 3=Medium, 4=Low, so a guessed number "
            "silently sets the wrong value. Never set yourself as an issue's "
            "delegate or assignee unless the user explicitly asks you to. "
            "When your work "
            "produces an external artifact (a PR, document, or dashboard), "
            "attach it with linear_agent_set_session_links using the Session "
            "ID from your prompt. For multi-step work, mirror your todo plan "
            "into linear_agent_update_plan (send the FULL step list every time, "
            "updating each step's status as it starts and finishes) so Linear "
            "renders live progress. Never write tool-call syntax (e.g. "
            "<tool_code>) into your response text; always emit real tool calls."
        ),
    )
