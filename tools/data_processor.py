"""
Data Processor Tool for JARVIS
==============================

Provides comprehensive data transformation and analysis capabilities.
"""

import asyncio
import json
import csv
import yaml
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import statistics
import re
from pathlib import Path

from .base import BaseTool, ToolMetadata, ToolCategory


class DataProcessorTool(BaseTool):
    """
    Advanced data processing and transformation tool

    Features:
    - Multiple format support (JSON, CSV, YAML, etc.)
    - Data transformation operations
    - Statistical analysis
    - Data validation and cleaning
    - Aggregation and grouping
    - Format conversion
    """

    def __init__(self):
        metadata = ToolMetadata(
            name="data_processor",
            description="Process, transform, and analyze structured data",
            category=ToolCategory.DATA,
            version="1.0.0",
            tags=["data", "transform", "analysis", "processing"],
            required_permissions=["data_access"],
            rate_limit=None,
            timeout=60,
            examples=[
                {
                    "description": "Transform JSON data",
                    "params": {
                        "operation": "transform",
                        "data": [
                            {"name": "John", "age": 30},
                            {"name": "Jane", "age": 25},
                        ],
                        "transformations": [
                            {
                                "type": "add_field",
                                "field": "adult",
                                "value": "lambda x: x['age'] >= 18",
                            }
                        ],
                    },
                },
                {
                    "description": "Calculate statistics",
                    "params": {
                        "operation": "statistics",
                        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "metrics": ["mean", "median", "std_dev"],
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Supported operations
        self.operations = {
            "transform": self._handle_transform,
            "filter": self._handle_filter,
            "aggregate": self._handle_aggregate,
            "statistics": self._handle_statistics,
            "convert": self._handle_convert,
            "validate": self._handle_validate,
            "clean": self._handle_clean,
            "merge": self._handle_merge,
            "sort": self._handle_sort,
            "pivot": self._handle_pivot,
        }

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate data processing parameters"""
        operation = kwargs.get("operation")

        if not operation:
            return False, "Operation parameter is required"

        if operation not in self.operations:
            return False, f"Invalid operation: {operation}"

        data = kwargs.get("data")
        if data is None:
            return False, "Data parameter is required"

        # Operation-specific validation
        if operation == "merge":
            if "merge_data" not in kwargs:
                return False, "merge_data parameter required for merge operation"

        if operation == "statistics":
            if not isinstance(data, (list, tuple)) or not data:
                return False, "Statistics operation requires non-empty list of numbers"

        return True, None

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data processing operation"""
        operation = kwargs.get("operation")
        handler = self.operations.get(operation)

        if not handler:
            raise ValueError(f"Unknown operation: {operation}")

        return await handler(**kwargs)

    async def _handle_transform(self, **kwargs) -> Dict[str, Any]:
        """Transform data with various operations"""
        data = kwargs.get("data")
        transformations = kwargs.get("transformations", [])

        # Convert to DataFrame for easier manipulation
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({"value": data})

        original_shape = df.shape

        # Apply transformations
        for transform in transformations:
            transform_type = transform.get("type")

            if transform_type == "add_field":
                field_name = transform.get("field")
                value = transform.get("value")

                if isinstance(value, str) and value.startswith("lambda"):
                    # Evaluate lambda expression (careful with security)
                    try:
                        func = eval(value)
                        df[field_name] = df.apply(func, axis=1)
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Lambda evaluation failed: {e}",
                        }
                else:
                    df[field_name] = value

            elif transform_type == "rename_field":
                old_name = transform.get("old_name")
                new_name = transform.get("new_name")
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})

            elif transform_type == "drop_field":
                field_name = transform.get("field")
                if field_name in df.columns:
                    df = df.drop(columns=[field_name])

            elif transform_type == "convert_type":
                field_name = transform.get("field")
                target_type = transform.get("target_type")
                if field_name in df.columns:
                    try:
                        if target_type == "int":
                            df[field_name] = df[field_name].astype(int)
                        elif target_type == "float":
                            df[field_name] = df[field_name].astype(float)
                        elif target_type == "str":
                            df[field_name] = df[field_name].astype(str)
                        elif target_type == "datetime":
                            df[field_name] = pd.to_datetime(df[field_name])
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Type conversion failed: {e}",
                        }

        # Convert back to original format
        if len(df.columns) == 1 and "value" in df.columns:
            result_data = df["value"].tolist()
        else:
            result_data = df.to_dict("records")

        return {
            "operation": "transform",
            "original_shape": original_shape,
            "final_shape": df.shape,
            "transformations_applied": len(transformations),
            "data": result_data,
            "columns": df.columns.tolist(),
        }

    async def _handle_filter(self, **kwargs) -> Dict[str, Any]:
        """Filter data based on conditions"""
        data = kwargs.get("data")
        conditions = kwargs.get("conditions", [])
        mode = kwargs.get("mode", "and")  # "and" or "or"

        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({"value": data})

        original_count = len(df)

        # Apply filters
        if conditions:
            masks = []

            for condition in conditions:
                field = condition.get("field")
                operator = condition.get("operator")
                value = condition.get("value")

                if field not in df.columns:
                    continue

                if operator == "eq":
                    mask = df[field] == value
                elif operator == "ne":
                    mask = df[field] != value
                elif operator == "gt":
                    mask = df[field] > value
                elif operator == "gte":
                    mask = df[field] >= value
                elif operator == "lt":
                    mask = df[field] < value
                elif operator == "lte":
                    mask = df[field] <= value
                elif operator == "contains":
                    mask = df[field].astype(str).str.contains(str(value), na=False)
                elif operator == "in":
                    mask = df[field].isin(value if isinstance(value, list) else [value])
                else:
                    continue

                masks.append(mask)

            if masks:
                if mode == "and":
                    final_mask = pd.concat(masks, axis=1).all(axis=1)
                else:  # or
                    final_mask = pd.concat(masks, axis=1).any(axis=1)

                df = df[final_mask]

        # Convert back to original format
        if len(df.columns) == 1 and "value" in df.columns:
            result_data = df["value"].tolist()
        else:
            result_data = df.to_dict("records")

        return {
            "operation": "filter",
            "original_count": original_count,
            "filtered_count": len(df),
            "removed_count": original_count - len(df),
            "conditions_applied": len(conditions),
            "mode": mode,
            "data": result_data,
        }

    async def _handle_aggregate(self, **kwargs) -> Dict[str, Any]:
        """Aggregate data with grouping and functions"""
        data = kwargs.get("data")
        group_by = kwargs.get("group_by", [])
        aggregations = kwargs.get("aggregations", {})

        if not isinstance(data, list) or not all(
            isinstance(item, dict) for item in data
        ):
            return {
                "success": False,
                "error": "Aggregate requires list of dictionaries",
            }

        df = pd.DataFrame(data)

        # Perform aggregation
        if group_by:
            grouped = df.groupby(group_by)

            agg_dict = {}
            for field, funcs in aggregations.items():
                if field in df.columns:
                    if isinstance(funcs, str):
                        funcs = [funcs]
                    agg_dict[field] = funcs

            if agg_dict:
                result = grouped.agg(agg_dict)
                result = result.reset_index()

                # Flatten column names
                result.columns = [
                    f"{col[0]}_{col[1]}" if col[1] else col[0]
                    for col in result.columns.values
                ]
            else:
                # Default aggregation
                result = grouped.size().reset_index(name="count")
        else:
            # Global aggregation
            result_dict = {}
            for field, funcs in aggregations.items():
                if field in df.columns:
                    if isinstance(funcs, str):
                        funcs = [funcs]
                    for func in funcs:
                        if func == "sum":
                            result_dict[f"{field}_sum"] = df[field].sum()
                        elif func == "mean":
                            result_dict[f"{field}_mean"] = df[field].mean()
                        elif func == "count":
                            result_dict[f"{field}_count"] = df[field].count()
                        elif func == "min":
                            result_dict[f"{field}_min"] = df[field].min()
                        elif func == "max":
                            result_dict[f"{field}_max"] = df[field].max()

            result = pd.DataFrame([result_dict])

        return {
            "operation": "aggregate",
            "group_by": group_by,
            "aggregations": aggregations,
            "result_shape": result.shape,
            "data": result.to_dict("records"),
        }

    async def _handle_statistics(self, **kwargs) -> Dict[str, Any]:
        """Calculate statistical metrics"""
        data = kwargs.get("data")
        metrics = kwargs.get("metrics", ["mean", "median", "std_dev"])

        # Flatten data if needed
        if isinstance(data, list) and all(
            isinstance(item, (int, float)) for item in data
        ):
            values = data
        else:
            return {"success": False, "error": "Statistics requires list of numbers"}

        if not values:
            return {"success": False, "error": "Empty data"}

        results = {"operation": "statistics", "count": len(values), "metrics": {}}

        # Calculate requested metrics
        if "mean" in metrics:
            results["metrics"]["mean"] = statistics.mean(values)

        if "median" in metrics:
            results["metrics"]["median"] = statistics.median(values)

        if "mode" in metrics:
            try:
                results["metrics"]["mode"] = statistics.mode(values)
            except statistics.StatisticsError:
                results["metrics"]["mode"] = None

        if "std_dev" in metrics:
            if len(values) > 1:
                results["metrics"]["std_dev"] = statistics.stdev(values)
            else:
                results["metrics"]["std_dev"] = 0

        if "variance" in metrics:
            if len(values) > 1:
                results["metrics"]["variance"] = statistics.variance(values)
            else:
                results["metrics"]["variance"] = 0

        if "min" in metrics:
            results["metrics"]["min"] = min(values)

        if "max" in metrics:
            results["metrics"]["max"] = max(values)

        if "range" in metrics:
            results["metrics"]["range"] = max(values) - min(values)

        if "sum" in metrics:
            results["metrics"]["sum"] = sum(values)

        if "percentiles" in metrics:
            results["metrics"]["percentiles"] = {
                "25": np.percentile(values, 25),
                "50": np.percentile(values, 50),
                "75": np.percentile(values, 75),
                "90": np.percentile(values, 90),
                "95": np.percentile(values, 95),
                "99": np.percentile(values, 99),
            }

        return results

    async def _handle_convert(self, **kwargs) -> Dict[str, Any]:
        """Convert data between formats"""
        data = kwargs.get("data")
        from_format = kwargs.get("from_format", "auto")
        to_format = kwargs.get("to_format", "json")

        # Auto-detect format if needed
        if from_format == "auto":
            if isinstance(data, str):
                if data.strip().startswith("{") or data.strip().startswith("["):
                    from_format = "json"
                elif "," in data and "\n" in data:
                    from_format = "csv"
                else:
                    from_format = "text"
            elif isinstance(data, (list, dict)):
                from_format = "python"

        # Parse input data
        if from_format == "json" and isinstance(data, str):
            parsed_data = json.loads(data)
        elif from_format == "csv" and isinstance(data, str):
            reader = csv.DictReader(data.splitlines())
            parsed_data = list(reader)
        elif from_format == "yaml" and isinstance(data, str):
            parsed_data = yaml.safe_load(data)
        else:
            parsed_data = data

        # Convert to target format
        if to_format == "json":
            if isinstance(parsed_data, (list, dict)):
                result = json.dumps(parsed_data, indent=2)
            else:
                result = json.dumps({"data": parsed_data}, indent=2)

        elif to_format == "csv":
            if isinstance(parsed_data, list) and all(
                isinstance(item, dict) for item in parsed_data
            ):
                output = []
                if parsed_data:
                    keys = parsed_data[0].keys()
                    output.append(",".join(keys))
                    for item in parsed_data:
                        output.append(",".join(str(item.get(k, "")) for k in keys))
                result = "\n".join(output)
            else:
                result = str(parsed_data)

        elif to_format == "yaml":
            result = yaml.dump(parsed_data, default_flow_style=False)

        elif to_format == "markdown":
            if isinstance(parsed_data, list) and all(
                isinstance(item, dict) for item in parsed_data
            ):
                # Create markdown table
                if parsed_data:
                    keys = list(parsed_data[0].keys())
                    result = "| " + " | ".join(keys) + " |\n"
                    result += "| " + " | ".join(["---"] * len(keys)) + " |\n"
                    for item in parsed_data:
                        result += (
                            "| "
                            + " | ".join(str(item.get(k, "")) for k in keys)
                            + " |\n"
                        )
                else:
                    result = "No data"
            else:
                result = f"```\n{parsed_data}\n```"

        else:
            result = str(parsed_data)

        return {
            "operation": "convert",
            "from_format": from_format,
            "to_format": to_format,
            "input_type": type(data).__name__,
            "output_type": type(result).__name__,
            "data": result,
        }

    async def _handle_validate(self, **kwargs) -> Dict[str, Any]:
        """Validate data against schema or rules"""
        data = kwargs.get("data")
        schema = kwargs.get("schema", {})
        rules = kwargs.get("rules", [])

        validation_results = {
            "operation": "validate",
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # Schema validation
        if schema:
            if isinstance(data, list):
                for i, item in enumerate(data):
                    errors = self._validate_against_schema(item, schema, f"item[{i}]")
                    validation_results["errors"].extend(errors)
            else:
                errors = self._validate_against_schema(data, schema, "data")
                validation_results["errors"].extend(errors)

        # Rule validation
        for rule in rules:
            rule_type = rule.get("type")

            if rule_type == "required_fields":
                fields = rule.get("fields", [])
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, dict):
                            missing = [f for f in fields if f not in item]
                            if missing:
                                validation_results["errors"].append(
                                    f"Item {i} missing required fields: {missing}"
                                )
                elif isinstance(data, dict):
                    missing = [f for f in fields if f not in data]
                    if missing:
                        validation_results["errors"].append(
                            f"Missing required fields: {missing}"
                        )

            elif rule_type == "unique_field":
                field = rule.get("field")
                if isinstance(data, list) and field:
                    values = [
                        item.get(field) for item in data if isinstance(item, dict)
                    ]
                    if len(values) != len(set(values)):
                        validation_results["errors"].append(
                            f"Field '{field}' contains duplicate values"
                        )

        validation_results["is_valid"] = len(validation_results["errors"]) == 0
        validation_results["error_count"] = len(validation_results["errors"])
        validation_results["warning_count"] = len(validation_results["warnings"])

        return validation_results

    async def _handle_clean(self, **kwargs) -> Dict[str, Any]:
        """Clean data by removing nulls, duplicates, etc."""
        data = kwargs.get("data")
        operations = kwargs.get("operations", ["remove_nulls", "remove_duplicates"])

        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({"value": data})

        original_shape = df.shape
        cleaning_log = []

        # Apply cleaning operations
        if "remove_nulls" in operations:
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                cleaning_log.append(f"Removed {before - after} rows with null values")

        if "remove_duplicates" in operations:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            if before != after:
                cleaning_log.append(f"Removed {before - after} duplicate rows")

        if "trim_strings" in operations:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.strip()
            cleaning_log.append("Trimmed whitespace from string fields")

        if "remove_empty_strings" in operations:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].replace("", np.nan)
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                cleaning_log.append(f"Removed {before - after} rows with empty strings")

        # Convert back to original format
        if len(df.columns) == 1 and "value" in df.columns:
            result_data = df["value"].tolist()
        else:
            result_data = df.to_dict("records")

        return {
            "operation": "clean",
            "original_shape": original_shape,
            "final_shape": df.shape,
            "operations_applied": operations,
            "cleaning_log": cleaning_log,
            "rows_removed": original_shape[0] - df.shape[0],
            "data": result_data,
        }

    async def _handle_merge(self, **kwargs) -> Dict[str, Any]:
        """Merge two datasets"""
        data = kwargs.get("data")
        merge_data = kwargs.get("merge_data")
        on = kwargs.get("on")  # Field(s) to merge on
        how = kwargs.get("how", "inner")  # inner, outer, left, right

        if not isinstance(data, list) or not isinstance(merge_data, list):
            return {"success": False, "error": "Both data and merge_data must be lists"}

        df1 = pd.DataFrame(data)
        df2 = pd.DataFrame(merge_data)

        # Perform merge
        if on:
            merged = pd.merge(df1, df2, on=on, how=how)
        else:
            # If no key specified, try to find common columns
            common_cols = list(set(df1.columns) & set(df2.columns))
            if common_cols:
                merged = pd.merge(df1, df2, on=common_cols[0], how=how)
            else:
                # No common columns, do a cross join
                df1["_temp_key"] = 1
                df2["_temp_key"] = 1
                merged = pd.merge(df1, df2, on="_temp_key", how="outer")
                merged = merged.drop("_temp_key", axis=1)

        return {
            "operation": "merge",
            "left_shape": df1.shape,
            "right_shape": df2.shape,
            "merged_shape": merged.shape,
            "merge_key": on,
            "merge_type": how,
            "data": merged.to_dict("records"),
        }

    async def _handle_sort(self, **kwargs) -> Dict[str, Any]:
        """Sort data"""
        data = kwargs.get("data")
        by = kwargs.get("by")  # Field(s) to sort by
        ascending = kwargs.get("ascending", True)

        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            if by and by in df.columns:
                df = df.sort_values(by=by, ascending=ascending)
            result_data = df.to_dict("records")
        elif isinstance(data, list):
            result_data = sorted(data, reverse=not ascending)
        else:
            return {"success": False, "error": "Data must be a list"}

        return {
            "operation": "sort",
            "sorted_by": by,
            "ascending": ascending,
            "count": len(result_data),
            "data": result_data,
        }

    async def _handle_pivot(self, **kwargs) -> Dict[str, Any]:
        """Pivot data table"""
        data = kwargs.get("data")
        index = kwargs.get("index")
        columns = kwargs.get("columns")
        values = kwargs.get("values")
        aggfunc = kwargs.get("aggfunc", "sum")

        if not isinstance(data, list) or not all(
            isinstance(item, dict) for item in data
        ):
            return {"success": False, "error": "Pivot requires list of dictionaries"}

        df = pd.DataFrame(data)

        # Create pivot table
        pivot = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0,
        )

        # Convert to records format
        pivot = pivot.reset_index()

        return {
            "operation": "pivot",
            "index": index,
            "columns": columns,
            "values": values,
            "aggfunc": aggfunc,
            "shape": pivot.shape,
            "data": pivot.to_dict("records"),
        }

    def _validate_against_schema(
        self, data: Any, schema: Dict[str, Any], path: str
    ) -> List[str]:
        """Validate data against a schema"""
        errors = []

        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                errors.append(f"{path} should be an object")
            elif expected_type == "array" and not isinstance(data, list):
                errors.append(f"{path} should be an array")
            elif expected_type == "string" and not isinstance(data, str):
                errors.append(f"{path} should be a string")
            elif expected_type == "number" and not isinstance(data, (int, float)):
                errors.append(f"{path} should be a number")
            elif expected_type == "boolean" and not isinstance(data, bool):
                errors.append(f"{path} should be a boolean")

        if "properties" in schema and isinstance(data, dict):
            for prop, prop_schema in schema["properties"].items():
                if prop in data:
                    errors.extend(
                        self._validate_against_schema(
                            data[prop], prop_schema, f"{path}.{prop}"
                        )
                    )
                elif prop_schema.get("required", False):
                    errors.append(f"{path}.{prop} is required")

        return errors

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Document the parameters for this tool"""
        return {
            "operation": {
                "type": "string",
                "description": "The data operation to perform",
                "required": True,
                "enum": list(self.operations.keys()),
            },
            "data": {
                "type": "any",
                "description": "The data to process",
                "required": True,
            },
            "transformations": {
                "type": "array",
                "description": "List of transformations for transform operation",
                "required": False,
            },
            "conditions": {
                "type": "array",
                "description": "Filter conditions",
                "required": False,
            },
            "group_by": {
                "type": "array",
                "description": "Fields to group by for aggregation",
                "required": False,
            },
            "aggregations": {
                "type": "object",
                "description": "Aggregation functions to apply",
                "required": False,
            },
            "metrics": {
                "type": "array",
                "description": "Statistical metrics to calculate",
                "required": False,
                "default": ["mean", "median", "std_dev"],
            },
        }


# Example usage
async def example_usage():
    """Example of using the DataProcessorTool"""
    tool = DataProcessorTool()

    # Example 1: Transform data
    result = await tool.execute(
        operation="transform",
        data=[
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "San Francisco"},
            {"name": "Charlie", "age": 35, "city": "New York"},
        ],
        transformations=[
            {
                "type": "add_field",
                "field": "adult",
                "value": "lambda x: x['age'] >= 18",
            },
            {
                "type": "add_field",
                "field": "age_group",
                "value": "lambda x: 'Young' if x['age'] < 30 else 'Adult'",
            },
        ],
    )
    print("Transform result:", json.dumps(result.data, indent=2))

    # Example 2: Calculate statistics
    result = await tool.execute(
        operation="statistics",
        data=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        metrics=["mean", "median", "std_dev", "percentiles"],
    )
    print("\nStatistics result:", json.dumps(result.data, indent=2))


if __name__ == "__main__":
    asyncio.run(example_usage())
