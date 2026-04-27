# Orbital Widgets API Reference

Version 4.2 — published March 2025.

## Authentication

All requests must include a bearer token in the `Authorization` header. Tokens are issued via the `/auth/token` endpoint and expire after **45 minutes**. Refresh tokens are valid for 14 days and are issued alongside the access token in the same response body under the `refresh_token` field.

When an access token expires, requests return HTTP `401` with the error code `auth.token_expired`. Clients are expected to call `/auth/refresh` with the refresh token to obtain a new access token. If the refresh token is also expired, the response is `401` with code `auth.refresh_required`, and the client must restart the OAuth flow.

Tokens issued before March 2025 used HS256; tokens issued from version 4.0 onward use RS256 with a public key published at `/.well-known/jwks.json`.

## Rate limits

The default plan allows **120 requests per minute** per API key, measured against a sliding window. The Pro plan raises this to **600 requests per minute**, and Enterprise plans are negotiated individually.

When the limit is exceeded, the response is HTTP `429` with the error code `rate.limit_exceeded`. The response includes a `Retry-After` header giving the number of seconds until the next request will be accepted.

## Pagination

Every list endpoint is paginated. Requests accept the query parameters `cursor` (opaque string) and `limit` (1–100, default **25**).

Paginated endpoints include:
- `GET /widgets` — list widgets in the workspace
- `GET /widgets/{id}/events` — list events for a widget
- `GET /workspaces/{id}/members` — list workspace members
- `GET /audit-log` — list audit events

Responses include a `next_cursor` field. When `next_cursor` is `null`, there are no more results.

## Error codes

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `request.invalid` | Malformed request body |
| 401 | `auth.token_expired` | Access token expired |
| 401 | `auth.refresh_required` | Refresh token expired or revoked |
| 403 | `permission.denied` | Caller lacks required scope |
| 404 | `resource.not_found` | Resource does not exist or is hidden from caller |
| 409 | `resource.conflict` | Resource already exists or is in an incompatible state |
| 429 | `rate.limit_exceeded` | Rate limit hit |
| 500 | `server.internal` | Unexpected server error |
| 503 | `server.maintenance` | Scheduled maintenance — retry after the time given in `Retry-After` |

## Webhooks

Webhooks deliver events with at-least-once semantics. Each delivery is signed with HMAC-SHA256 using the workspace's webhook secret; the signature appears in the `X-Orbital-Signature` header. Clients must respond with HTTP `2xx` within **10 seconds** or the delivery is retried with exponential backoff for up to 24 hours.
