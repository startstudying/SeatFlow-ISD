"""Contract tests for SeatFlow V3 JSON Schema definitions."""

import json
from pathlib import Path

import pytest

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "json-schema"

# Schemas whose "*_id" fields must be type string (identity fields).
IDENTITY_SCHEMAS = {
    "seat.schema.json",
    "stock.schema.json",
    "lhb-event.schema.json",
    "event-cycle.schema.json",
    "prediction-sample.schema.json",
}

ALL_SCHEMA_FILES = sorted(SCHEMAS_DIR.glob("*.schema.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_schema(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Draft 2020-12 compliance
# ---------------------------------------------------------------------------

class TestDraft202012:
    """Every schema must declare $schema as Draft 2020-12 and carry an $id
    rooted at https://seatflow.local/schemas/v3/."""

    @pytest.mark.parametrize("schema_path", ALL_SCHEMA_FILES, ids=lambda p: p.name)
    def test_dollar_schema_is_2020_12(self, schema_path: Path):
        schema = _load_schema(schema_path)
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", (
            f"{schema_path.name}: $schema must be Draft 2020-12"
        )

    @pytest.mark.parametrize("schema_path", ALL_SCHEMA_FILES, ids=lambda p: p.name)
    def test_dollar_id_under_seatflow_v3(self, schema_path: Path):
        schema = _load_schema(schema_path)
        dollar_id = schema.get("$id", "")
        assert dollar_id.startswith("https://seatflow.local/schemas/v3/"), (
            f"{schema_path.name}: $id must start with https://seatflow.local/schemas/v3/"
        )


# ---------------------------------------------------------------------------
# 2. Identity fields are strings
# ---------------------------------------------------------------------------

class TestIdentityFieldsAreStrings:
    """In the designated identity schemas, any property whose name ends with
    '_id' must be declared as type 'string' (not integer or number)."""

    @pytest.mark.parametrize(
        "schema_path",
        [p for p in ALL_SCHEMA_FILES if p.name in IDENTITY_SCHEMAS],
        ids=lambda p: p.name,
    )
    def test_id_fields_are_strings(self, schema_path: Path):
        schema = _load_schema(schema_path)
        properties = schema.get("properties", {})

        id_fields = {
            name: prop for name, prop in properties.items()
            if name.endswith("_id")
        }

        assert id_fields, f"{schema_path.name}: expected at least one '*_id' field"

        for field_name, field_def in id_fields.items():
            # The type must be "string" (or a list where "string" is the first entry).
            field_type = field_def.get("type")
            if isinstance(field_type, list):
                assert field_type[0] == "string", (
                    f"{schema_path.name}.{field_name}: first type must be 'string', "
                    f"got {field_type}"
                )
            else:
                assert field_type == "string", (
                    f"{schema_path.name}.{field_name}: type must be 'string', "
                    f"got {field_type}"
                )


# ---------------------------------------------------------------------------
# 3. Required fields list present and non-empty
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Every schema must have a non-empty 'required' array."""

    @pytest.mark.parametrize("schema_path", ALL_SCHEMA_FILES, ids=lambda p: p.name)
    def test_has_required_fields(self, schema_path: Path):
        schema = _load_schema(schema_path)
        required = schema.get("required")
        assert isinstance(required, list) and len(required) > 0, (
            f"{schema_path.name}: 'required' must be a non-empty list"
        )

    @pytest.mark.parametrize("schema_path", ALL_SCHEMA_FILES, ids=lambda p: p.name)
    def test_required_fields_exist_in_properties(self, schema_path: Path):
        schema = _load_schema(schema_path)
        properties = set(schema.get("properties", {}).keys())
        required = schema.get("required", [])
        missing = set(required) - properties
        assert not missing, (
            f"{schema_path.name}: required fields not found in properties: {missing}"
        )


# ---------------------------------------------------------------------------
# 4. additionalProperties: false
# ---------------------------------------------------------------------------

class TestForbidAdditionalProperties:
    """Every schema must set additionalProperties to false."""

    @pytest.mark.parametrize("schema_path", ALL_SCHEMA_FILES, ids=lambda p: p.name)
    def test_additional_properties_false(self, schema_path: Path):
        schema = _load_schema(schema_path)
        assert schema.get("additionalProperties") is False, (
            f"{schema_path.name}: additionalProperties must be false"
        )
