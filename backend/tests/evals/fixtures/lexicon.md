# Penumbra Labs Lexicon

A glossary of terms used across Penumbra Labs documentation.

## Quanto-mesh

A peer-to-peer routing layer used by Penumbra Labs sensors to forward telemetry when no direct uplink to the central aggregator is available. Each sensor maintains links to up to **eight** neighbors and forwards packets using a modified version of AODV. Quanto-mesh links are encrypted end-to-end with the same device certificates used by the RedShift protocol.

## Telemark relay

A high-availability hardware appliance that bridges Quanto-mesh networks to the public internet. A Telemark relay terminates RedShift sessions on the mesh side and re-establishes them against the central aggregator. Each relay can sustain up to **2,000 concurrent RedShift sessions** and is rated for outdoor deployment from -40°C to +60°C.

## Beacon Frame

The Quanto-mesh discovery message broadcast every **5 seconds** by every sensor. It carries the sensor's device ID, current battery level, and the number of mesh neighbors currently linked. Beacon frames are unsigned and unencrypted.

## Anchor Node

A sensor designated as an entry point for time synchronization. Anchor nodes have a wired GPS clock source and broadcast time updates to neighbors at 1 Hz. A Quanto-mesh requires at least one Anchor Node to maintain accurate timestamps; deployments without one fall back to a 30-second resync interval against the Telemark relay.

## Eclipse Window

A configurable interval during which a sensor suspends Beacon Frame broadcasts to conserve battery. The default Eclipse Window is **02:00–04:00 local time**. Sensors in an Eclipse Window still respond to direct queries but do not advertise themselves to new neighbors.

## Drift Coefficient

The measured ratio between a sensor's local clock and the network's reference clock, expressed in parts per million (ppm). Sensors with a Drift Coefficient above **50 ppm** are flagged for hardware inspection, because they typically indicate a failing crystal oscillator.
