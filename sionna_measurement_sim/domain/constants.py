"""Shared schema constants."""

SCHEMA_VERSION = "2.1.0"
FULL_CONTRACT_NAME = "sionna_measurement_sim_hdf5"
RT_LABELS_CONTRACT_NAME = "sionna_measurement_rt_labels"
IQ_LINK_LIBRARY_CONTRACT_NAME = "sionna_measurement_iq_link_library"
CONTRACT_NAME = FULL_CONTRACT_NAME
DEFAULT_OUTPUT_PROFILE = "full"
OUTPUT_PROFILES = ("full", "rt_lite", "rt_labels_only", "iq_link_library")
INDEX_ORDER = "tx,rx,rx_ant,tx_ant,..."
UNIT_CONVENTION = "si_mks"
PRODUCER = "sionna_measurement_sim"
