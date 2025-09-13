"""
Unit tests for conversation starters functionality.

Tests the flag key generation and conversation starter mapping functions.
"""

import unittest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from src.functions import flags_to_key, add_conversation_starters


class TestConversationStarters(unittest.TestCase):
    """Test conversation starter functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_conversation_starters = {
            "0000": "Test message for no flags",
            "1010": "Test message for central + fernwaerme", 
            "0101": "Test message for fossil + wucher"
        }
        
        self.test_gdf = gpd.GeoDataFrame({
            'central_heating_flag': [False, True, False, True],
            'fossil_heating_flag': [False, False, True, True], 
            'fernwaerme_flag': [False, True, False, False],
            'wucher_miete_flag': [False, False, True, False],
            'renter_flag': [True, True, True, True],
            'geometry': [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(4)]
        }, crs='EPSG:4326')

    def test_flags_to_key_basic(self):
        """Test basic flag key generation."""
        test_cases = [
            (False, False, False, False, "0000"),
            (True, False, False, False, "1000"),
            (False, True, False, False, "0100"),
            (False, False, True, False, "0010"),
            (False, False, False, True, "0001"),
            (True, True, True, True, "1111")
        ]
        
        for central, fossil, fernwaerme, wucher, expected in test_cases:
            with self.subTest(expected=expected):
                row = pd.Series({
                    'central_heating_flag': central,
                    'fossil_heating_flag': fossil,
                    'fernwaerme_flag': fernwaerme,
                    'wucher_miete_flag': wucher
                })
                result = flags_to_key(row)
                self.assertEqual(result, expected)

    def test_flags_to_key_with_actual_data(self):
        """Test flag key generation with real data."""
        for i, row in self.test_gdf.iterrows():
            flag_key = flags_to_key(row)
            # Should be 4 characters
            self.assertEqual(len(flag_key), 4)
            # Should only contain 0s and 1s
            self.assertTrue(all(c in '01' for c in flag_key))

    def test_add_conversation_starters_basic(self):
        """Test basic conversation starter addition."""
        result = add_conversation_starters(self.test_gdf, self.test_conversation_starters)
        
        # Should have original columns plus 2 new ones
        expected_cols = len(self.test_gdf.columns) + 2
        self.assertEqual(len(result.columns), expected_cols)
        
        # Should have flag_key and conversation_start columns
        self.assertIn('flag_key', result.columns)
        self.assertIn('conversation_start', result.columns)
        
        # All rows should have flag keys and conversation starters
        self.assertTrue(result['flag_key'].notna().all())
        self.assertTrue(result['conversation_start'].notna().all())

    def test_add_conversation_starters_mapping(self):
        """Test correct mapping of conversation starters."""
        result = add_conversation_starters(self.test_gdf, self.test_conversation_starters)
        
        # First row: 0000 -> should map to test message
        first_row = result.iloc[0]
        self.assertEqual(first_row['flag_key'], '0000')
        self.assertEqual(first_row['conversation_start'], "Test message for no flags")
        
        # Second row: 1010 -> should map to test message
        second_row = result.iloc[1] 
        self.assertEqual(second_row['flag_key'], '1010')
        self.assertEqual(second_row['conversation_start'], "Test message for central + fernwaerme")

    def test_add_conversation_starters_fallback(self):
        """Test fallback behavior for unmapped keys."""
        # Use empty conversation starters dict to force fallback
        result = add_conversation_starters(self.test_gdf, {})
        
        default_message = "Hallo, ich bin von der Linken. Wie geht es Ihnen mit den Wohn- und Nebenkosten?"
        
        # All rows should use the fallback message
        for _, row in result.iterrows():
            self.assertEqual(row['conversation_start'], default_message)

    def test_add_conversation_starters_preserves_original_data(self):
        """Test that original data is preserved."""
        original_cols = list(self.test_gdf.columns)
        result = add_conversation_starters(self.test_gdf, self.test_conversation_starters)
        
        # All original columns should still be present
        for col in original_cols:
            self.assertIn(col, result.columns)
            # Data should be identical
            pd.testing.assert_series_equal(self.test_gdf[col], result[col])

    def test_add_conversation_starters_returns_copy(self):
        """Test that function returns a copy, not modifying original."""
        original_shape = self.test_gdf.shape
        result = add_conversation_starters(self.test_gdf, self.test_conversation_starters)
        
        # Original should be unchanged
        self.assertEqual(self.test_gdf.shape, original_shape)
        self.assertNotIn('flag_key', self.test_gdf.columns)
        self.assertNotIn('conversation_start', self.test_gdf.columns)
        
        # Result should be different
        self.assertNotEqual(result.shape, original_shape)


if __name__ == '__main__':
    unittest.main()
