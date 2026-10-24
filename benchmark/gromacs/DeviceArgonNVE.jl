# The device engine moved into the package (src/engine/, ParticleDynamics.DeviceEngine).
# Benchmarks and tests that `include` this file keep their `DeviceArgonNVE` name.
const DeviceArgonNVE = ParticleDynamics.DeviceEngine
