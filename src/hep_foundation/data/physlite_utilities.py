import json
import logging
from enum import Enum
from importlib import resources
from typing import Any, Optional

# Import functions to check and retrieve derived feature definitions
from hep_foundation.data.physlite_derived_features import (
    get_derived_feature,
    is_derived_feature,
)

logger = logging.getLogger(__name__)

# Load the branch index data from the JSON file
try:
    # Use importlib.resources for robust path finding within the package
    # The path should be relative to the package root (hep_foundation)
    json_path_str = str(
        resources.files("hep_foundation").joinpath("data/physlite_branch_index.json")
    )
    with (
        resources.files("hep_foundation")
        .joinpath("data/physlite_branch_index.json")
        .open("rt") as f
    ):
        data = json.load(f)
        PHYSLITE_BRANCHES = data["physlite_branches"]
        logger.info(f"Successfully loaded PhysLite branch index from {json_path_str}")
        # Optionally log metadata:
        # generation_info = data.get("generation_info", {})
        # logger.info(f"Branch index generated from: {generation_info}")
except FileNotFoundError:
    logger.error(
        "physlite_branch_index.json not found. Please generate it using the script in scripts/."
    )
    PHYSLITE_BRANCHES = {}
except (json.JSONDecodeError, KeyError) as e:
    logger.error(f"Error parsing physlite_branch_index.json: {e}")
    PHYSLITE_BRANCHES = {}


class BranchType(Enum):
    """Enum for branch types based on their shape."""

    UNKNOWN = "unknown"
    FEATURE = "feature"  # Scalar value (single value per event)
    FEATURE_ARRAY = "feature_array"  # Array value (multiple values per event)


def get_branch_info(
    branch_name: str,
) -> tuple[bool, BranchType, Optional[dict[str, Any]]]:
    """
    Check if a branch name is valid (either real or derived) and determine its type.

    Args:
        branch_name: Full branch name (e.g., "InDetTrackParticlesAuxDyn.d0" or "derived.XYZ")

    Returns:
        Tuple containing:
        - Boolean indicating if branch exists (or is a known derived feature)
        - BranchType enum value based on shape (real or derived)
        - Dictionary with branch information (from index or constructed for derived)

    Raises:
        RuntimeError: If branch index is not available and the branch is not derived.
    """
    # First, check if it's a known derived feature
    if is_derived_feature(branch_name):
        derived_feature = get_derived_feature(branch_name)
        if derived_feature:
            # Get the constructed branch info dict (shape, dtype, status)
            branch_info_dict = derived_feature.get_branch_info_dict()
            # Determine the branch type from the derived feature's shape
            branch_type = _determine_branch_type(branch_info_dict)
            logger.info(
                f"Branch '{branch_name}' identified as derived: type={branch_type}, info={branch_info_dict}"
            )
            return True, branch_type, branch_info_dict
        else:
            # Should not happen if is_derived_feature is True, but handle defensively
            logger.warning(
                f"Branch '{branch_name}' flagged as derived but definition not found."
            )
            return False, BranchType.UNKNOWN, None

    # If not derived, proceed to check the PhysLite index file
    if not PHYSLITE_BRANCHES:
        # Raise error only if it's not derived and the index is missing
        raise RuntimeError(
            "PhysLite branch index not found and branch is not derived. Cannot validate branches. "
            "Please ensure the branch index is properly installed and accessible."
        )

    # Extract category and feature name for index lookup
    branch_name_str: str = branch_name  # Ensure it's a string
    category: Optional[str] = None
    feature: str = branch_name_str

    if "." in branch_name_str:
        category, feature = branch_name_str.split(".", 1)

    # Check if the branch exists in the loaded dictionary
    branch_info: Optional[dict[str, Any]] = None
    if category:
        if category in PHYSLITE_BRANCHES and feature in PHYSLITE_BRANCHES[category]:
            branch_info = PHYSLITE_BRANCHES[category][feature]
    else:  # Assume 'Other' category if no dot
        if "Other" in PHYSLITE_BRANCHES and feature in PHYSLITE_BRANCHES["Other"]:
            branch_info = PHYSLITE_BRANCHES["Other"][feature]

    if not branch_info:
        # It wasn't derived and wasn't found in the index
        logger.warning(
            f"Branch '{branch_name}' not found in PHYSLITE_BRANCHES index and is not a known derived feature."
        )
        return False, BranchType.UNKNOWN, None

    # Check status (only for branches found in the index)
    status = branch_info.get("status", "unknown")
    if status != "success":
        logger.warning(
            f"Branch '{branch_name}' has status '{status}' in index. Treating as invalid."
        )
        return False, BranchType.UNKNOWN, None

    # Determine branch type from shape information (only for branches found in the index)
    branch_type = _determine_branch_type(branch_info)

    # If we reached here, it's a valid branch from the index file
    return True, branch_type, branch_info


def _determine_branch_type(branch_info: dict[str, Any]) -> BranchType:
    """Helper function to determine branch type from its info."""
    if "shape" not in branch_info or branch_info["shape"] is None:
        return BranchType.UNKNOWN

    # Convert string representation of shape to tuple if needed
    shape = branch_info["shape"]
    if isinstance(shape, str):
        try:
            # Handle tuple string like "()" or "(10,)"
            shape = eval(shape)
        except Exception:
            return BranchType.UNKNOWN

    # Empty tuple or tuple with zeros indicates a scalar (feature)
    if not shape or shape == () or shape == (0,):
        return BranchType.FEATURE

    # Non-empty shape indicates an array (feature_array)
    return BranchType.FEATURE_ARRAY


class PhysliteBranch:
    """
    Represents a validated branch in the PhysLite data.

    Attributes:
        name: The full branch name
        category: The category part of the branch name
        feature: The feature part of the branch name
        branch_type: The type of branch (feature or feature_array)
        info: Additional information about the branch
    """

    def __init__(self, branch_name: str):
        """
        Initialize a PhysliteBranch object.

        Args:
            branch_name: Full branch name (e.g., "InDetTrackParticlesAuxDyn.d0")

        Raises:
            ValueError: If the branch name is invalid
        """
        is_valid, branch_type, branch_info = get_branch_info(branch_name)

        if not is_valid:
            raise ValueError(f"Invalid branch name: {branch_name}")

        self.name = branch_name
        self.branch_type = branch_type
        self.info = branch_info

        # Split the branch name into category and feature
        if "." in branch_name:
            self.category, self.feature = branch_name.split(".", 1)
        else:
            self.category = "Other"
            self.feature = branch_name

    @property
    def is_feature(self) -> bool:
        """Check if this branch is a scalar feature (single value per event)."""
        return self.branch_type == BranchType.FEATURE

    @property
    def is_feature_array(self) -> bool:
        """Check if this branch is a feature array (multiple values per event)."""
        return self.branch_type == BranchType.FEATURE_ARRAY

    def __str__(self) -> str:
        return f"PhysliteBranch({self.name}, type={self.branch_type.value})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_shape(self) -> Optional[tuple[int, ...]]:
        """Get the shape of this branch if available."""
        if not self.info or "shape" not in self.info:
            return None

        shape = self.info["shape"]
        if isinstance(shape, str):
            try:
                # Handle tuple string like "()" or "(10,)"
                shape = eval(shape)
            except Exception:
                return None

        return shape


class PhysliteFeatureSelector:
    """
    Selector for scalar PhysLite features (single value per event).

    Attributes:
        branch: The PhysliteBranch to select
    """

    def __init__(self, branch: PhysliteBranch):
        """
        Initialize a feature selector.

        Args:
            branch: PhysliteBranch to select

        Raises:
            ValueError: If branch is not a scalar feature type
        """
        if not branch.is_feature:
            raise ValueError(
                f"Branch {branch.name} is not a scalar feature. Use PhysliteFeatureArraySelector instead."
            )

        self.branch = branch

    def __str__(self) -> str:
        return f"FeatureSelector({self.branch.name})"

    def __repr__(self) -> str:
        return self.__str__()


class PhysliteFeatureArraySelector:
    """
    Selector for PhysLite feature arrays (multiple values per event).

    Attributes:
        branch: The PhysliteBranch to select
    """

    def __init__(self, branch: PhysliteBranch):
        """
        Initialize a feature array selector.

        Args:
            branch: PhysliteBranch to select

        Raises:
            ValueError: If branch is not a feature array type
        """
        if not branch.is_feature_array:
            raise ValueError(
                f"Branch {branch.name} is not a feature array. Use PhysliteFeatureSelector instead."
            )

        self.branch = branch

    def __str__(self) -> str:
        return f"FeatureArraySelector({self.branch.name})"

    def __repr__(self) -> str:
        return self.__str__()


class PhysliteFeatureFilter:
    """
    Filter for scalar PhysLite features (single value per event).

    Attributes:
        branch: The PhysliteBranch to filter on
        min_value: Minimum allowed value (None means no minimum)
        max_value: Maximum allowed value (None means no maximum)
    """

    def __init__(
        self,
        branch: PhysliteBranch,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        Initialize a feature filter.

        Args:
            branch: PhysliteBranch to filter on
            min_value: Minimum allowed value (None means no minimum)
            max_value: Maximum allowed value (None means no maximum)

        Raises:
            ValueError: If branch is not a scalar feature type or if both min_value and max_value are None
        """
        if not branch.is_feature:
            raise ValueError(
                f"Branch {branch.name} is not a scalar feature. Use PhysliteFeatureArrayFilter instead."
            )

        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")

        self.branch = branch
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self) -> str:
        min_str = str(self.min_value) if self.min_value is not None else "-∞"
        max_str = str(self.max_value) if self.max_value is not None else "∞"
        return f"FeatureFilter({self.branch.name}, range=[{min_str}, {max_str}])"

    def __repr__(self) -> str:
        return self.__str__()


class PhysliteFeatureArrayFilter:
    """
    Filter for PhysLite feature arrays (multiple values per event).

    Attributes:
        branch: The PhysliteBranch to filter on
        min_value: Minimum allowed value (None means no minimum)
        max_value: Maximum allowed value (None means no maximum)
    """

    def __init__(
        self,
        branch: PhysliteBranch,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        Initialize a feature array filter.

        Args:
            branch: PhysliteBranch to filter on
            min_value: Minimum allowed value (None means no minimum)
            max_value: Maximum allowed value (None means no maximum)

        Raises:
            ValueError: If branch is not a feature array type or if both min_value and max_value are None
        """
        if not branch.is_feature_array:
            raise ValueError(
                f"Branch {branch.name} is not a feature array. Use PhysliteFeatureFilter instead."
            )

        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")

        self.branch = branch
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self) -> str:
        min_str = str(self.min_value) if self.min_value is not None else "-∞"
        max_str = str(self.max_value) if self.max_value is not None else "∞"
        return f"FeatureArrayFilter({self.branch.name}, range=[{min_str}, {max_str}])"

    def __repr__(self) -> str:
        return self.__str__()


class PhysliteFeatureArrayAggregator:
    """
    Configuration for aggregating multiple feature arrays from PhysLite data.

    This class bundles configuration parameters for extracting, filtering,
    and aggregating feature arrays from PhysLite events.

    Attributes:
        input_branches: List of feature array selectors for input data collection
        filter_branches: List of feature array filters for filtering data
        sort_by_branch: Feature array selector to use for sorting (typically pT), or None to keep original order
        min_length: Minimum number of array elements required after filtering
        max_length: Maximum number of array elements to keep (truncation/padding size)
    """

    def __init__(
        self,
        input_branches: list[PhysliteFeatureArraySelector],
        filter_branches: list[PhysliteFeatureArrayFilter],
        sort_by_branch: Optional[PhysliteFeatureArraySelector] = None,
        min_length: int = 1,
        max_length: int = 100,
    ):
        """
        Initialize a feature array aggregator configuration.

        Args:
            input_branches: List of feature array selectors to extract
            filter_branches: List of feature array filters to apply filtering
            sort_by_branch: Feature array selector to use for sorting values, or None to keep original order
            min_length: Minimum number of elements required after filtering
            max_length: Maximum number of elements to keep (will pad or truncate)

        Raises:
            ValueError: If input parameters are invalid
        """
        if not input_branches:
            raise ValueError("At least one input branch must be provided")

        if min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {min_length}")

        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        if min_length > max_length:
            raise ValueError(
                f"min_length ({min_length}) cannot be greater than max_length ({max_length})"
            )

        self.input_branches = input_branches
        self.filter_branches = filter_branches
        self.sort_by_branch = sort_by_branch
        self.min_length = min_length
        self.max_length = max_length

        # Get branch names for easier reference
        self.input_branch_names = [f.branch.name for f in input_branches]
        self.filter_branch_names = [f.branch.name for f in filter_branches]
        self.sort_by_branch_name = (
            sort_by_branch.branch.name if sort_by_branch is not None else None
        )

    def __str__(self) -> str:
        input_str = ", ".join(self.input_branch_names)
        filter_str = (
            ", ".join(self.filter_branch_names) if self.filter_branches else "none"
        )
        sort_str = (
            self.sort_by_branch_name if self.sort_by_branch_name is not None else "none"
        )
        return (
            f"FeatureArrayAggregator(branches=[{input_str}], "
            f"filters=[{filter_str}], "
            f"sort_by={sort_str}, "
            f"length={self.min_length}-{self.max_length})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class PhysliteSelectionConfig:
    """
    High-level configuration for PhysLite data selection and feature extraction.

    This class bundles together all parameters needed to define what features
    to extract from PhysLite data processing. Filtering is handled separately.

    Attributes:
        feature_selectors: List of scalar feature selectors for individual values
        feature_array_aggregators: List of feature array aggregators for collecting arrays
        name: Optional name for this configuration
    """

    def __init__(
        self,
        feature_selectors: list[PhysliteFeatureSelector] = None,
        feature_array_aggregators: list[PhysliteFeatureArrayAggregator] = None,
        name: str = "PhysliteSelection",
    ):
        """
        Initialize a PhysLite selection configuration.

        Args:
            feature_selectors: List of scalar feature selectors for individual values
            feature_array_aggregators: List of feature array aggregators for collecting arrays
            name: Optional name for this configuration

        Raises:
            ValueError: If no selectors or aggregators are provided
        """
        self.feature_selectors = feature_selectors or []
        self.feature_array_aggregators = feature_array_aggregators or []
        self.name = name

        # Ensure at least one selector or aggregator is provided
        if not (self.feature_selectors or self.feature_array_aggregators):
            raise ValueError(
                "At least one feature selector or aggregator must be provided"
            )

    def __str__(self) -> str:
        return (
            f"PhysliteSelectionConfig(name='{self.name}', "
            f"feature_selectors={len(self.feature_selectors)}, "
            f"feature_array_aggregators={len(self.feature_array_aggregators)})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def get_total_feature_size(self) -> int:
        """
        Calculate total feature size by combining scalar features and aggregated array features.

        Returns:
            int: Total number of features, calculated as:
                 (number of scalar features) +
                 sum(aggregator.max_length * len(aggregator.input_branches) for each aggregator)
        """
        # Count scalar features
        total_size = len(self.feature_selectors)

        # Add size from each aggregator
        for aggregator in self.feature_array_aggregators:
            # Calculate size contributed by this aggregator's inputs
            aggregator_feature_count_per_track = 0
            for selector in aggregator.input_branches:
                # --- Determine feature multiplicity (k) from branch shape ---
                branch_shape = selector.branch.get_shape()
                k = 1  # Default to 1 feature
                if branch_shape is not None:
                    if len(branch_shape) == 1:
                        # Shape like (-1,) or (N,) -> k=1 feature per track
                        k = 1
                    elif len(branch_shape) == 2:
                        # Shape like (-1, k) or (N, k) -> k features per track
                        # Use the second dimension as k
                        k = branch_shape[1]
                        if k <= 0:
                            # Handle cases like shape [-1, 0] or [-1, -1] if they occur
                            logger.warning(
                                f"Branch '{selector.branch.name}' has non-positive inner dimension {k} in shape {branch_shape}. Assuming k=1."
                            )
                            k = 1
                    else:
                        # Unexpected shape dimensions (e.g., 0D or 3D+ for an aggregator input)
                        logger.warning(
                            f"Branch '{selector.branch.name}' has unexpected shape {branch_shape} for feature size calculation. Assuming k=1."
                        )
                        k = 1
                else:
                    logger.warning(
                        f"Branch '{selector.branch.name}' has no shape info available. Assuming k=1."
                    )

                aggregator_feature_count_per_track += k

            # Multiply total features per track by max_length for this aggregator
            aggregator_size = aggregator.max_length * aggregator_feature_count_per_track
            total_size += aggregator_size

        return total_size
