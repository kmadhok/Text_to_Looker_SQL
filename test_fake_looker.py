#!/usr/bin/env python3
"""
Unit tests for fake_looker.py to ensure SDK parity and proper validation.
Tests cover all main SDK methods, query processing, joins, filters, and error handling.
"""

import pytest
import json
import pandas as pd
from fake_looker import FakeLookerSDK, all_lookml_models, lookml_model, lookml_model_explore, run_inline_query

class TestFakeLookerSDK:
    """Test suite for the fake Looker SDK."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with SDK instance."""
        cls.sdk = FakeLookerSDK()
    
    def test_all_lookml_models(self):
        """Test all_lookml_models function."""
        models = all_lookml_models()
        assert isinstance(models, list)
        assert len(models) == 1
        assert models[0]["name"] == "retail"
    
    def test_lookml_model(self):
        """Test lookml_model function."""
        model = lookml_model("retail")
        assert model["name"] == "retail"
        assert "explores" in model
        assert len(model["explores"]) == 5
        
        explore_names = [exp["name"] for exp in model["explores"]]
        expected_explores = ["customers", "products", "stores", "orders", "order_items"]
        for exp in expected_explores:
            assert exp in explore_names
    
    def test_lookml_model_invalid(self):
        """Test lookml_model with invalid model name."""
        try:
            lookml_model("invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model 'invalid' not found" in str(e)
    
    def test_lookml_model_explore(self):
        """Test lookml_model_explore function."""
        explore = lookml_model_explore("retail", "customers")
        
        assert explore["name"] == "customers"
        assert explore["label"] == "Customers"
        assert "fields" in explore
        assert "dimensions" in explore["fields"]
        assert "measures" in explore["fields"]
        
        # Check specific fields exist
        dimension_names = [dim["name"] for dim in explore["fields"]["dimensions"]]
        assert "customer_id" in dimension_names
        assert "first_name" in dimension_names
        assert "state" in dimension_names
        
        measure_names = [meas["name"] for meas in explore["fields"]["measures"]]
        assert "count" in measure_names
    
    def test_lookml_model_explore_invalid(self):
        """Test lookml_model_explore with invalid inputs."""
        try:
            lookml_model_explore("invalid", "customers")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model 'invalid' not found" in str(e)
        
        try:
            lookml_model_explore("retail", "invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Explore 'invalid' not found" in str(e)
    
    def test_simple_query(self):
        """Test simple query execution."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id", "first_name", "state"],
            "limit": 10
        }
        
        result = run_inline_query("json", query)
        data = json.loads(result)
        
        assert isinstance(data, list)
        assert len(data) == 10
        
        # Check first row has expected fields
        first_row = data[0]
        assert "customer_id" in first_row
        assert "first_name" in first_row
        assert "state" in first_row
    
    def test_measure_query(self):
        """Test query with measures."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["count", "avg_age"],
            "limit": 1
        }
        
        result = run_inline_query("json", query)
        data = json.loads(result)
        
        assert isinstance(data, list)
        assert len(data) == 1
        
        row = data[0]
        assert "count" in row
        assert "avg_age" in row
        assert row["count"] == 5000  # Total customer count
        assert isinstance(row["avg_age"], float)
    
    def test_query_with_filters(self):
        """Test query with filters."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id", "state"],
            "filters": {"state": "CA"},
            "limit": 100
        }
        
        result = run_inline_query("json", query)
        data = json.loads(result)
        
        # All results should be from CA
        for row in data:
            assert row["state"] == "CA"
    
    def test_query_with_numeric_filter(self):
        """Test query with numeric filters."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id", "age"],
            "filters": {"age": ">50"},
            "limit": 100
        }
        
        result = run_inline_query("json", query)
        data = json.loads(result)
        
        # All results should have age > 50
        for row in data:
            assert row["age"] > 50
    
    def test_query_with_sorts(self):
        """Test query with sorting."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id", "age"],
            "sorts": ["age desc"],
            "limit": 5
        }
        
        result = run_inline_query("json", query)
        data = json.loads(result)
        
        # Results should be sorted by age descending
        ages = [row["age"] for row in data]
        assert ages == sorted(ages, reverse=True)
    
    def test_query_validation_missing_fields(self):
        """Test query validation for missing required fields."""
        query = {
            "model": "retail",
            "explore": "customers"
            # Missing "fields"
        }
        
        try:
            run_inline_query("json", query)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Missing required field: fields" in str(e)
    
    def test_query_validation_invalid_model(self):
        """Test query validation for invalid model."""
        query = {
            "model": "invalid",
            "explore": "customers",
            "fields": ["customer_id"]
        }
        
        try:
            run_inline_query("json", query)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model 'invalid' not found" in str(e)
    
    def test_query_validation_invalid_explore(self):
        """Test query validation for invalid explore."""
        query = {
            "model": "retail",
            "explore": "invalid",
            "fields": ["customer_id"]
        }
        
        try:
            run_inline_query("json", query)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Explore 'invalid' not found" in str(e)
    
    def test_query_validation_invalid_field(self):
        """Test query validation for invalid field."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["invalid_field"]
        }
        
        try:
            run_inline_query("json", query)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Field 'invalid_field' not found" in str(e)
    
    def test_unsupported_pivots(self):
        """Test that pivots raise NotImplementedError."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id"],
            "pivots": ["state"]
        }
        
        try:
            run_inline_query("json", query)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError as e:
            assert "Pivots are not supported" in str(e)
    
    def test_unsupported_totals(self):
        """Test that totals raise NotImplementedError."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id"],
            "total": True
        }
        
        try:
            run_inline_query("json", query)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError as e:
            assert "Totals are not supported" in str(e)
    
    def test_csv_output_format(self):
        """Test CSV output format."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id", "first_name"],
            "limit": 5
        }
        
        result = run_inline_query("csv", query)
        assert isinstance(result, str)
        assert "customer_id,first_name" in result
        lines = result.strip().split('\n')
        assert len(lines) == 6  # Header + 5 data rows
    
    def test_unsupported_result_format(self):
        """Test unsupported result format."""
        query = {
            "model": "retail",
            "explore": "customers",
            "fields": ["customer_id"],
            "limit": 1
        }
        
        try:
            run_inline_query("xml", query)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported result format: xml" in str(e)
    
    def test_order_items_with_measures(self):
        """Test order_items explore with complex measures."""
        query = {
            "model": "retail",
            "explore": "order_items",
            "fields": ["total_quantity", "avg_item_price", "total_sales"],
            "limit": 1
        }
        
        result = run_inline_query("json", query)
        data = json.loads(result)
        
        assert len(data) == 1
        row = data[0]
        
        assert "total_quantity" in row
        assert "avg_item_price" in row
        assert "total_sales" in row
        assert isinstance(row["total_quantity"], (int, float))
        assert isinstance(row["avg_item_price"], (int, float))
        assert isinstance(row["total_sales"], (int, float))

if __name__ == "__main__":
    # Run tests directly
    import sys
    
    print("🧪 Running Fake Looker SDK Tests...")
    
    # Create test instance
    test_suite = TestFakeLookerSDK()
    test_suite.setup_class()
    
    # List of all test methods
    test_methods = [
        test_suite.test_all_lookml_models,
        test_suite.test_lookml_model,
        test_suite.test_lookml_model_invalid,
        test_suite.test_lookml_model_explore,
        test_suite.test_lookml_model_explore_invalid,
        test_suite.test_simple_query,
        test_suite.test_measure_query,
        test_suite.test_query_with_filters,
        test_suite.test_query_with_numeric_filter,
        test_suite.test_query_with_sorts,
        test_suite.test_query_validation_missing_fields,
        test_suite.test_query_validation_invalid_model,
        test_suite.test_query_validation_invalid_explore,
        test_suite.test_query_validation_invalid_field,
        test_suite.test_unsupported_pivots,
        test_suite.test_unsupported_totals,
        test_suite.test_csv_output_format,
        test_suite.test_unsupported_result_format,
        test_suite.test_order_items_with_measures,
    ]
    
    passed = 0
    failed = 0
    
    # Run all tests
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 All tests passed!")
        sys.exit(0) 