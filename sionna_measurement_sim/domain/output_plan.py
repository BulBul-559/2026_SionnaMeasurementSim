"""Output product planning for RT truth pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sionna_measurement_sim.domain.constants import (
    FULL_CONTRACT_NAME,
    IQ_LINK_LIBRARY_CONTRACT_NAME,
    OUTPUT_PRODUCT_ALL,
    OUTPUT_PRODUCT_ARRAY,
    OUTPUT_PRODUCT_CALIBRATION,
    OUTPUT_PRODUCT_CFR_OBS,
    OUTPUT_PRODUCT_CFR_TRUTH,
    OUTPUT_PRODUCT_CIR_TRUTH,
    OUTPUT_PRODUCT_DERIVED,
    OUTPUT_PRODUCT_IQ,
    OUTPUT_PRODUCT_MOTION,
    OUTPUT_PRODUCT_MULTIUSER,
    OUTPUT_PRODUCT_NLOS_PATH_TRUTH,
    OUTPUT_PRODUCT_PATH_FULL,
    OUTPUT_PRODUCT_PATH_SAMPLES,
    OUTPUT_PRODUCT_RANGING,
    OUTPUT_PRODUCT_VISUALIZATION,
    OUTPUT_PRODUCTS,
    OUTPUT_PROFILES,
    RT_LABELS_CONTRACT_NAME,
)

FULL_OUTPUT_PRODUCTS: tuple[str, ...] = (
    OUTPUT_PRODUCT_DERIVED,
    OUTPUT_PRODUCT_CFR_TRUTH,
    OUTPUT_PRODUCT_CIR_TRUTH,
    OUTPUT_PRODUCT_PATH_SAMPLES,
    OUTPUT_PRODUCT_NLOS_PATH_TRUTH,
    OUTPUT_PRODUCT_PATH_FULL,
    OUTPUT_PRODUCT_CFR_OBS,
    OUTPUT_PRODUCT_ARRAY,
    OUTPUT_PRODUCT_RANGING,
    OUTPUT_PRODUCT_IQ,
    OUTPUT_PRODUCT_MULTIUSER,
    OUTPUT_PRODUCT_CALIBRATION,
    OUTPUT_PRODUCT_MOTION,
    OUTPUT_PRODUCT_VISUALIZATION,
)

RT_LITE_PRODUCTS: tuple[str, ...] = (
    OUTPUT_PRODUCT_DERIVED,
    OUTPUT_PRODUCT_CFR_TRUTH,
    OUTPUT_PRODUCT_CIR_TRUTH,
    OUTPUT_PRODUCT_PATH_SAMPLES,
    OUTPUT_PRODUCT_NLOS_PATH_TRUTH,
    OUTPUT_PRODUCT_MOTION,
)

PRODUCT_ALIASES: dict[str, str] = {
    "cfr_observation": OUTPUT_PRODUCT_CFR_OBS,
    "observation": OUTPUT_PRODUCT_CFR_OBS,
    "obs": OUTPUT_PRODUCT_CFR_OBS,
    "rtt": OUTPUT_PRODUCT_RANGING,
    "range": OUTPUT_PRODUCT_RANGING,
    "spectrum": OUTPUT_PRODUCT_ARRAY,
    "spatial_spectrum": OUTPUT_PRODUCT_ARRAY,
    "paths": OUTPUT_PRODUCT_PATH_SAMPLES,
    "full_paths": OUTPUT_PRODUCT_PATH_FULL,
}

PHY_PRODUCTS = frozenset(
    {
        OUTPUT_PRODUCT_CALIBRATION,
        OUTPUT_PRODUCT_CFR_OBS,
        OUTPUT_PRODUCT_RANGING,
        OUTPUT_PRODUCT_IQ,
        OUTPUT_PRODUCT_MULTIUSER,
    }
)
OBSERVATION_ARRAY_SOURCES = frozenset({"cfr_est", "rx_grid"})


@dataclass(frozen=True)
class RTOutputPlan:
    """Concrete compute/write plan derived from output profile/products."""

    profile: str = "full"
    contract_name: str = FULL_CONTRACT_NAME
    products: tuple[str, ...] = FULL_OUTPUT_PRODUCTS
    array_sources: tuple[str, ...] = ()
    compute_cfr: bool = True
    compute_cir: bool = True
    compute_path_samples: bool = True
    compute_nlos_truth: bool = True
    write_full_contract: bool = True
    write_compact_link_labels: bool = False
    write_iq_link_library: bool = False

    @property
    def is_custom_products(self) -> bool:
        return self.profile == "custom"

    @property
    def requires_phy_observation(self) -> bool:
        if set(self.products) & PHY_PRODUCTS:
            return True
        return (
            OUTPUT_PRODUCT_ARRAY in self.products
            and bool(set(self.array_sources) & OBSERVATION_ARRAY_SOURCES)
        )

    @property
    def write_derived(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_DERIVED)

    @property
    def write_cfr_truth(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_CFR_TRUTH)

    @property
    def write_cir_truth(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_CIR_TRUTH)

    @property
    def write_path_samples(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_PATH_SAMPLES)

    @property
    def write_nlos_path_truth(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_NLOS_PATH_TRUTH)

    @property
    def write_path_full(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_PATH_FULL)

    @property
    def write_cfr_observation(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_CFR_OBS)

    @property
    def write_array_outputs(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_ARRAY)

    @property
    def write_ranging(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_RANGING)

    @property
    def write_iq(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_IQ)

    @property
    def write_multiuser(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_MULTIUSER)

    @property
    def write_calibration(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_CALIBRATION)

    @property
    def write_motion(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_MOTION)

    @property
    def write_visualization(self) -> bool:
        return self._writes(OUTPUT_PRODUCT_VISUALIZATION)

    def _writes(self, product: str) -> bool:
        return product in self.products


def build_rt_output_plan(
    profile: str,
    products: Iterable[str] | None = None,
    array_sources: Iterable[str] | None = None,
) -> RTOutputPlan:
    """Return the concrete RT output plan for a validated profile/product set."""

    profile = str(profile)
    if profile not in OUTPUT_PROFILES:
        allowed = ", ".join(OUTPUT_PROFILES)
        msg = f"output.profile must be one of: {allowed}"
        raise ValueError(msg)

    if products is not None:
        if profile in ("rt_labels_only", "iq_link_library"):
            msg = "output.products cannot be combined with compact output profiles"
            raise ValueError(msg)
        normalized = _normalize_products(products)
        return _plan_for_products(
            normalized,
            profile="custom",
            array_sources=array_sources,
        )

    if profile == "custom":
        msg = "output.profile='custom' requires output.products"
        raise ValueError(msg)
    if profile == "rt_labels_only":
        return RTOutputPlan(
            profile=profile,
            contract_name=RT_LABELS_CONTRACT_NAME,
            products=(),
            compute_cfr=False,
            compute_cir=False,
            compute_path_samples=False,
            compute_nlos_truth=False,
            write_full_contract=False,
            write_compact_link_labels=True,
        )
    if profile == "iq_link_library":
        return RTOutputPlan(
            profile=profile,
            contract_name=IQ_LINK_LIBRARY_CONTRACT_NAME,
            products=(OUTPUT_PRODUCT_IQ,),
            compute_cfr=True,
            compute_cir=False,
            compute_path_samples=False,
            compute_nlos_truth=False,
            write_full_contract=False,
            write_iq_link_library=True,
        )
    if profile == "rt_lite":
        return _plan_for_products(
            RT_LITE_PRODUCTS,
            profile=profile,
            array_sources=array_sources,
        )
    return _plan_for_products(
        FULL_OUTPUT_PRODUCTS,
        profile=profile,
        array_sources=array_sources,
    )


def _plan_for_products(
    products: tuple[str, ...],
    *,
    profile: str,
    array_sources: Iterable[str] | None = None,
) -> RTOutputPlan:
    product_set = set(products)
    resolved_array_sources = tuple(str(source) for source in (array_sources or ()))
    compute_cfr = bool(
        product_set
        & {
            OUTPUT_PRODUCT_CFR_TRUTH,
            OUTPUT_PRODUCT_CFR_OBS,
            OUTPUT_PRODUCT_ARRAY,
            OUTPUT_PRODUCT_RANGING,
            OUTPUT_PRODUCT_IQ,
            OUTPUT_PRODUCT_MULTIUSER,
            OUTPUT_PRODUCT_CALIBRATION,
        }
    )
    compute_cir = OUTPUT_PRODUCT_CIR_TRUTH in product_set
    compute_path_samples = OUTPUT_PRODUCT_PATH_SAMPLES in product_set
    compute_nlos_truth = OUTPUT_PRODUCT_NLOS_PATH_TRUTH in product_set
    return RTOutputPlan(
        profile=profile,
        products=products,
        array_sources=resolved_array_sources,
        compute_cfr=compute_cfr,
        compute_cir=compute_cir,
        compute_path_samples=compute_path_samples,
        compute_nlos_truth=compute_nlos_truth,
    )


def _normalize_products(products: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in products:
        product = PRODUCT_ALIASES.get(str(raw), str(raw))
        if product == OUTPUT_PRODUCT_ALL:
            product_values = FULL_OUTPUT_PRODUCTS
        else:
            product_values = (product,)
        for value in product_values:
            if value not in OUTPUT_PRODUCTS or value == OUTPUT_PRODUCT_ALL:
                allowed = ", ".join(OUTPUT_PRODUCTS)
                msg = f"Unknown output product {value!r}; allowed values: {allowed}"
                raise ValueError(msg)
            if value not in normalized:
                normalized.append(value)
    if not normalized:
        msg = "output.products must contain at least one product"
        raise ValueError(msg)
    return tuple(normalized)
