# RedShift Protocol Specification

Current version: **2.1** (ratified by the RedShift Working Group, January 2026).

## Overview

RedShift is a binary application-layer protocol designed for low-latency telemetry from constrained devices (sensors, embedded radios) to a central aggregator. It runs over TCP or QUIC and uses a length-prefixed framing scheme.

## Version history

- **v1.0** (2022) — Initial release. Three message types: `GREET`, `DATA`, `BYE`. Plain TCP only.
- **v1.1** (2023) — Added `HEARTBEAT` for liveness; no breaking changes.
- **v2.0** (2025) — Major revision. Added QUIC transport, replaced `BYE` with `CLOSE` (carries reason code), introduced authenticated handshake. Backwards-incompatible.
- **v2.1** (2026) — Added optional `RESUME` message for fast reconnection using a session ticket from the previous `CLOSE`.

## Message types (v2.x)

- `GREET` — sent by client first, carries protocol version and supported cipher suites.
- `CHALLENGE` — server response to `GREET`, carries a 32-byte server nonce.
- `AUTH` — client signs the server nonce with its device key, returns the signature plus its certificate.
- `READY` — server's signal that the handshake completed; carries the negotiated cipher.
- `DATA` — telemetry payload. Body format is opaque to the protocol; routing is controlled by a 16-bit `topic_id`.
- `HEARTBEAT` — sent every 30 seconds when no `DATA` has been transmitted; one-way, no reply required.
- `RESUME` — optional, replaces the `GREET`/`CHALLENGE`/`AUTH` exchange when the client has a valid session ticket.
- `CLOSE` — terminates the session. Includes a one-byte reason code and may include a session ticket for future `RESUME` use.

In v1.x the only message types were `GREET`, `DATA`, `BYE`, and (from v1.1) `HEARTBEAT`. Authentication was out of scope.

## Handshake (v2.x)

The full handshake is four messages:

1. **GREET** — client → server. Announces protocol version and the cipher suites the client supports.
2. **CHALLENGE** — server → client. Server picks a cipher from the client's offered list and includes a 32-byte random nonce.
3. **AUTH** — client → server. Client computes Ed25519 signature over the nonce using its device key and sends the signature plus its X.509 device certificate.
4. **READY** — server → client. Server validates the certificate against the trust anchor, verifies the signature, and replies `READY` with the chosen cipher confirmed.

If any step fails, the server sends `CLOSE` with reason code `0x10` (auth failure) and disconnects.

## Reason codes (CLOSE message)

- `0x00` — normal shutdown
- `0x01` — protocol version not supported
- `0x10` — authentication failure
- `0x11` — certificate revoked
- `0x20` — rate limit exceeded
- `0xF0` — internal server error
