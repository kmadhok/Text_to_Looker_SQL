#!/usr/bin/env python3
"""
Fake Looker SDK that mimics the real Looker API.
Provides the same methods but operates on local CSV data using pandas.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
import re
from datetime import datetime, timedelta

class FakeLookerSDK:
    """Fake Looker SDK that operates on local CSV data."""
    
    def __init__(self):
        """Initialize the fake SDK and load data."""
        self.metadata = self._load_metadata()
        # Build in-memory map of join relationships directly from metadata so that
        # the stub behaves like the real Looker API (which serialises joins that
        # were defined in LookML).
        self.join_map = self._build_join_map()
        self.data_cache = {}
        self._load_data()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load the fake metadata from JSON file."""
        try:
            with open('fake_metadata.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("fake_metadata.json not found. Run make_synthetic_data.py first.")
    
    def _load_data(self):
        """Load all CSV data into memory for fast access."""
        table_files = {
            'customers': 'data/customers.csv',
            'products': 'data/products.csv', 
            'stores': 'data/stores.csv',
            'orders': 'data/orders.csv',
            'order_items': 'data/order_items.csv'
        }
        
        for table_name, file_path in table_files.items():
            try:
                self.data_cache[table_name] = pd.read_csv(file_path)
                # Convert date columns to datetime
                date_cols = [col for col in self.data_cache[table_name].columns if 'date' in col.lower()]
                for col in date_cols:
                    self.data_cache[table_name][col] = pd.to_datetime(self.data_cache[table_name][col])
            except FileNotFoundError:
                raise FileNotFoundError(f"Data file {file_path} not found. Run make_synthetic_data.py first.")
    
    def all_lookml_models(self, fields: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all LookML models (mimics real Looker API)."""
        return [{"name": self.metadata["model"]}]
    
    def lookml_model(self, model_name: str, fields: Optional[str] = None) -> Dict[str, Any]:
        """Return a specific LookML model with its explores."""
        if model_name != self.metadata["model"]:
            raise ValueError(f"Model '{model_name}' not found")
        
        return {
            "name": self.metadata["model"],
            "explores": [{"name": explore["name"]} for explore in self.metadata["explores"]]
        }
    
    def lookml_model_explore(self, model_name: str, explore_name: str, fields: Optional[str] = None) -> Dict[str, Any]:
        """Return a specific explore with its field definitions."""
        if model_name != self.metadata["model"]:
            raise ValueError(f"Model '{model_name}' not found")
        
        explore = None
        for exp in self.metadata["explores"]:
            if exp["name"] == explore_name:
                explore = exp
                break
        
        if not explore:
            raise ValueError(f"Explore '{explore_name}' not found")
        
        return explore
    
    def run_inline_query(self, result_format: str, write_query: Dict[str, Any]) -> str:
        """Execute a query and return results in specified format."""
        if result_format not in ["json", "csv"]:
            raise ValueError(f"Unsupported result format: {result_format}")
        
        # Validate query structure
        self._validate_query(write_query)
        
        # Get the base explore
        explore_name = write_query["model"] + ":" + write_query["explore"]
        base_table = write_query["explore"]
        
        if base_table not in self.data_cache:
            raise ValueError(f"Table '{base_table}' not found")
        
        # Start with the base table
        df = self.data_cache[base_table].copy()
        
        # Parse and apply joins based on selected fields
        df = self._apply_joins(df, write_query, base_table)
        
        # Apply filters
        df = self._apply_filters(df, write_query)
        
        # Select and compute fields
        df = self._select_fields(df, write_query)
        
        # Apply sorts
        df = self._apply_sorts(df, write_query)
        
        # Apply limit
        limit = write_query.get("limit", 500)
        df = df.head(limit)
        
        # Return in requested format
        if result_format == "json":
            return df.to_json(orient="records", date_format="iso")
        else:  # csv
            return df.to_csv(index=False)
    
    def _validate_query(self, query: Dict[str, Any]):
        """Validate the query structure and fields."""
        required_fields = ["model", "explore", "fields"]
        for field in required_fields:
            if field not in query:
                raise ValueError(f"Missing required field: {field}")
        
        # Check for unsupported features
        if "pivots" in query and query["pivots"]:
            raise NotImplementedError("Pivots are not supported")
        
        if "total" in query and query["total"]:
            raise NotImplementedError("Totals are not supported")
        
        # Validate model and explore exist
        if query["model"] != self.metadata["model"]:
            raise ValueError(f"Model '{query['model']}' not found")
        
        explore_names = [exp["name"] for exp in self.metadata["explores"]]
        if query["explore"] not in explore_names:
            raise ValueError(f"Explore '{query['explore']}' not found")
    
    def _find_existing_column(self, columns, base_key: str):
        """Return the actual column name that matches the logical key.
        Accepts either an exact match (base_key) or any column that has the
        form "{base_key}_<something>" (suffix added when a table was joined).
        Returns None if no candidate is found.
        """
        if base_key in columns:
            return base_key
        for col in columns:
            if col.startswith(f"{base_key}_") or col.endswith(f"_{base_key}"):
                return col
        return None

    def _apply_joins(self, df: pd.DataFrame, query: Dict[str, Any], base_table: str) -> pd.DataFrame:
        """Apply necessary joins based on selected fields.
        Enhanced to respect join dependency order – if a join relies on a key that
        appears only after another join (e.g. orders ➜ order_items ➜ products)
        we keep iterating until all required tables are joined or no further
        progress can be made.
        """
        # Parse field names to determine needed joins
        field_tables = set()
        for field_name in query["fields"]:
            if "." in field_name:
                field_tables.add(field_name.split(".")[0])
            else:
                field_tables.add(base_table)
        # We don't need to join the base table to itself
        field_tables.discard(base_table)

        remaining = set(field_tables)
        # Protect against infinite loops – no explore has more than, say, 10 joins
        max_iterations = 10
        while remaining and max_iterations > 0:
            progressed = False
            for table in list(remaining):
                if table not in self.data_cache:
                    continue  # Skip unknown tables
                join_key = self._get_join_key(base_table, table, self.join_map)
                if not join_key:
                    # Try to find an indirect path via an already-joined table
                    # e.g. base=orders target=products but we already joined order_items
                    for intermediate in field_tables | {base_table}:
                        if intermediate == table:
                            continue
                        potential = self._get_join_key(intermediate, table, self.join_map)
                        if potential and self._find_existing_column(df.columns, potential[0]):
                            # Treat intermediate as the new base for this step
                            potential_left_key, potential_right_key = potential
                            right_df = self.data_cache[table].copy()
                            right_df = right_df.add_suffix(f"_{table}")
                            right_df = right_df.rename(columns={f"{potential_right_key}_{table}": potential_right_key})
                            left_key_actual = self._find_existing_column(df.columns, potential_left_key)
                            df = pd.merge(df, right_df, left_on=left_key_actual, right_on=potential_right_key, how="left")
                            progressed = True
                            remaining.discard(table)
                            break
                    continue

                # Direct join – but only if the left key is already in df
                left_key, right_key = join_key if isinstance(join_key, tuple) else join_key
                left_key_actual = self._find_existing_column(df.columns, left_key)
                if not left_key_actual:
                    continue  # Wait until left key appears or is discoverable
                right_df = self.data_cache[table].copy()
                right_df = right_df.add_suffix(f"_{table}")
                right_df = right_df.rename(columns={f"{right_key}_{table}": right_key})
                df = pd.merge(df, right_df, left_on=left_key_actual, right_on=right_key, how="left", suffixes=("", f"_{table}"))
                progressed = True
                remaining.discard(table)
            if not progressed:
                # Could not resolve remaining joins – break to avoid infinite loop
                break
            max_iterations -= 1
        return df
    
    def _get_join_key(self, base_table: str, target_table: str, relationships: Dict) -> Optional[tuple]:
        """Get the join key between two tables."""
        # Check direct relationships
        key = (base_table, target_table)
        if key in relationships:
            return relationships[key]
        
        # Check reverse relationships  
        reverse_key = (target_table, base_table)
        if reverse_key in relationships:
            rel = relationships[reverse_key]
            if isinstance(rel, tuple):
                return (rel[1], rel[0])  # Swap keys
            return rel
        
        return None
    
    def _apply_filters(self, df: pd.DataFrame, query: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataframe."""
        filters = query.get("filters", {})
        
        # Handle case where filters is None (from Pydantic)
        if filters is None:
            filters = {}
        
        for field_name, filter_value in filters.items():
            if field_name in df.columns:
                df = self._apply_single_filter(df, field_name, filter_value)
        
        return df
    
    def _apply_single_filter(self, df: pd.DataFrame, field_name: str, filter_value: str) -> pd.DataFrame:
        """Apply a single filter to the dataframe."""
        # Handle date filters like "30 days"
        if "days" in filter_value.lower():
            days_match = re.search(r'(\d+)\s*days?', filter_value.lower())
            if days_match:
                days = int(days_match.group(1))
                cutoff_date = datetime.now() - timedelta(days=days)
                return df[df[field_name] >= cutoff_date]
        
        # Handle numeric comparisons
        if filter_value.startswith(('>', '<', '>=', '<=')):
            operator = re.match(r'(>=|<=|>|<)', filter_value).group(1)
            value_str = filter_value[len(operator):].strip()
            try:
                value = float(value_str)
                if operator == '>':
                    return df[df[field_name] > value]
                elif operator == '<':
                    return df[df[field_name] < value]
                elif operator == '>=':
                    return df[df[field_name] >= value]
                elif operator == '<=':
                    return df[df[field_name] <= value]
            except ValueError:
                pass  # Fall through to exact match
        
        # Exact match
        return df[df[field_name] == filter_value]
    
    def _select_fields(self, df: pd.DataFrame, query: Dict[str, Any]) -> pd.DataFrame:
        """Select and compute the requested fields."""
        field_list = query["fields"]
        result_df = pd.DataFrame()
        
        # Get field definitions
        explore_name = query["explore"]
        explore = None
        for exp in self.metadata["explores"]:
            if exp["name"] == explore_name:
                explore = exp
                break
        
        if not explore:
            raise ValueError(f"Explore '{explore_name}' not found")
        
        # Create field lookup
        all_fields = {}
        for dimension in explore["fields"]["dimensions"]:
            all_fields[dimension["name"]] = dimension
        for measure in explore["fields"]["measures"]:
            all_fields[measure["name"]] = measure
        
        # Process each requested field
        for field_name in field_list:
            # Handle table.field notation
            if "." in field_name:
                table, field = field_name.split(".", 1)
                # Look up field in the appropriate explore
                for exp in self.metadata["explores"]:
                    if exp["name"] == table:
                        table_fields = {}
                        for dim in exp["fields"]["dimensions"]:
                            table_fields[dim["name"]] = dim
                        for meas in exp["fields"]["measures"]:
                            table_fields[meas["name"]] = meas
                        
                        if field in table_fields:
                            field_def = table_fields[field]
                            result_df[field_name] = self._compute_field(df, field_def, field_name)
                            break
                else:
                    # Field not found in any explore
                    raise ValueError(f"Field '{field_name}' not found")
            else:
                # Look in current explore
                if field_name in all_fields:
                    field_def = all_fields[field_name]
                    result_df[field_name] = self._compute_field(df, field_def, field_name)
                else:
                    raise ValueError(f"Field '{field_name}' not found")
        
        return result_df
    
    def _compute_field(self, df: pd.DataFrame, field_def: Dict[str, Any], field_name: str) -> pd.Series:
        """Compute a field value based on its definition."""
        field_type = field_def["type"]
        sql = field_def["sql"]
        col_name = sql.replace("${TABLE}.", "")

        # Determine the potential suffixed column name in the joined DataFrame
        table = field_name.split('.')[0] if '.' in field_name else ''
        suffixed_col_name = f"{col_name}_{table}"

        target_col = None
        if col_name in df.columns:
            # Field is from the base table, not suffixed
            target_col = col_name
        elif suffixed_col_name in df.columns:
            # Field is from a joined table and has been suffixed
            target_col = suffixed_col_name
        else:
            raise ValueError(f"Column for field '{field_name}' (looking for '{col_name}' or '{suffixed_col_name}') not found in DataFrame.")

        if field_def["category"] == "dimension":
            return df[target_col]

        elif field_def["category"] == "measure":
            # NOTE: This aggregation is performed over the entire DataFrame,
            # as grouping is not yet implemented.
            if field_type == "count":
                return pd.Series([len(df)], index=[0])
            
            agg_map = {"sum": "sum", "average": "mean"}
            if field_type in agg_map:
                agg_func = agg_map[field_type]
                # The result of the aggregation is a single value; we broadcast it across a Series
                agg_value = df[target_col].agg(agg_func)
                return pd.Series([agg_value] * len(df.index), name=field_name)
            else:
                raise ValueError(f"Unsupported measure type: {field_type}")
        
        else:
            raise ValueError(f"Unknown field category: {field_def['category']}")
    
    def _apply_sorts(self, df: pd.DataFrame, query: Dict[str, Any]) -> pd.DataFrame:
        """Apply sorting to the dataframe."""
        sorts = query.get("sorts", [])
        
        # Handle case where sorts is None (from Pydantic)
        if sorts is None:
            sorts = []
            
        if not sorts:
            return df
        
        sort_columns = []
        sort_ascending = []
        
        for sort in sorts:
            if sort.endswith(" desc"):
                col_name = sort[:-5]
                ascending = False
            elif sort.endswith(" asc"):
                col_name = sort[:-4] 
                ascending = True
            else:
                col_name = sort
                ascending = True
            
            if col_name in df.columns:
                sort_columns.append(col_name)
                sort_ascending.append(ascending)
        
        if sort_columns:
            df = df.sort_values(by=sort_columns, ascending=sort_ascending)
        
        return df

    def _build_join_map(self):
        """Return {(base_view, joined_view): (base_col, joined_col)} dict derived from metadata."""
        join_map = {}
        for explore in self.metadata.get("explores", []):
            base = explore["name"]
            for join in explore.get("joins", []):
                target = join["name"]
                sql_on = join.get("sql_on", "")
                # Expect pattern like "${orders.customer_id} = ${customers.customer_id} ;;"
                try:
                    left, right = sql_on.split("=")
                    # Strip LookML syntax (${view.field}) and trailing characters
                    left_col = left.strip().lstrip("${").rstrip("}").split(".")[1]
                    right_col = right.strip().rstrip(" ;;").lstrip("${").rstrip("}").split(".")[1]
                    join_map[(base, target)] = (left_col, right_col)
                except Exception:
                    # If parsing fails just skip; developer can improve regex later
                    pass
        return join_map


# Global instance for easy access
sdk = FakeLookerSDK()

# Expose the main functions for compatibility
def all_lookml_models(fields=None):
    """Get all LookML models."""
    return sdk.all_lookml_models(fields)

def lookml_model(model_name, fields=None):
    """Get a specific LookML model."""
    return sdk.lookml_model(model_name, fields)

def lookml_model_explore(model_name, explore_name, fields=None):
    """Get a specific explore from a model."""
    return sdk.lookml_model_explore(model_name, explore_name, fields)

def run_inline_query(result_format, write_query):
    """Run an inline query and return results."""
    return sdk.run_inline_query(result_format, write_query) 