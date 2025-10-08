"""
Consolidated test suite for utility functions using pytest framework.
Tests JSON preprocessing and other utility functions.
"""
import pytest
import json
from src.utils import safe_json_loads, auto_close_json, preprocess_json_escapes, extract_json_object


class TestJSONUtilities:
    """Test cases for JSON utility functions."""
    
    def test_auto_close_json(self):
        """Test the auto_close_json function with various inputs."""
        test_cases = [
            # (input, expected_output)
            ('{"title": "Test", "summary": "This is a test', '{"title": "Test", "summary": "This is a test"}'),
            ('{"title": "Test", "summary": "This is a test"}', '{"title": "Test", "summary": "This is a test"}'),
            ('[{"item": 1}, {"item": 2}', '[{"item": 1}, {"item": 2}]'),
            ('{"data": {"items": [1, 2', '{"data": {"items": [1, 2]}}'),
            ('{"key": "value', '{"key": "value"}'),
            ('', ''),
            ('{"a": 1}', '{"a": 1}'),
            ('{"a": {"b": {"c": [1, 2, {"d": "e' , '{"a": {"b": {"c": [1, 2, {"d": "e"}]}}}')
        ]
        
        for input_str, expected_str in test_cases:
            result = auto_close_json(input_str)
            assert result == expected_str, f"Failed for input: {input_str}"
    
    def test_safe_json_loads_valid_json(self):
        """Test safe_json_loads with valid JSON."""
        valid_json = '{"title": "Test Meeting", "summary": "This is a test summary"}'
        result = safe_json_loads(valid_json)
        expected = {"title": "Test Meeting", "summary": "This is a test summary"}
        assert result == expected
    
    def test_safe_json_loads_with_markdown(self):
        """Test safe_json_loads with JSON wrapped in markdown code blocks."""
        markdown_json = '''```json
{
  "title": "Meeting Notes",
  "summary": "Key points discussed"
}
```'''
        result = safe_json_loads(markdown_json)
        expected = {"title": "Meeting Notes", "summary": "Key points discussed"}
        assert result == expected
    
    def test_safe_json_loads_with_unescaped_quotes(self):
        """Test safe_json_loads with unescaped quotes in string values."""
        malformed_json = '{"title": "John said "Hello world" to everyone", "summary": "Meeting summary"}'
        result = safe_json_loads(malformed_json)
        expected = {"title": 'John said "Hello world" to everyone', "summary": "Meeting summary"}
        assert result == expected
    
    def test_safe_json_loads_with_mixed_quotes(self):
        """Test safe_json_loads with mixed quote scenarios."""
        malformed_json = '{"title": "Alice\'s "big idea" presentation", "summary": "She said "this will change everything""}'
        result = safe_json_loads(malformed_json)
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result
    
    def test_safe_json_loads_with_newlines(self):
        """Test safe_json_loads with newlines and special characters."""
        malformed_json = '''{"title": "Complex Meeting", "summary": "Discussion about:\n- Point 1\n- Point 2 with "quotes"\n- Point 3"}'''
        result = safe_json_loads(malformed_json)
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result
    
    def test_safe_json_loads_empty_or_invalid_input(self):
        """Test safe_json_loads handling of empty or invalid input."""
        # Empty string
        result = safe_json_loads("", {"default": "value"})
        assert result == {"default": "value"}
        
        # None input
        result = safe_json_loads(None, {"default": "value"})
        assert result == {"default": "value"}
        
        # Non-string input
        result = safe_json_loads(123, {"default": "value"})
        assert result == {"default": "value"}
    
    def test_safe_json_loads_completely_malformed(self):
        """Test safe_json_loads with completely malformed JSON."""
        malformed_json = '{"title": "Test", "summary": unclosed string and missing quotes}'
        result = safe_json_loads(malformed_json, {"error": "fallback"})
        assert result == {"error": "fallback"}
    
    def test_safe_json_loads_array_format(self):
        """Test safe_json_loads with JSON array format."""
        json_array = '[{"speaker": "John", "sentence": "Hello everyone"}, {"speaker": "Jane", "sentence": "Good morning"}]'
        result = safe_json_loads(json_array)
        expected = [{"speaker": "John", "sentence": "Hello everyone"}, {"speaker": "Jane", "sentence": "Good morning"}]
        assert result == expected
    
    def test_preprocess_json_escapes_function(self):
        """Test the preprocess_json_escapes function directly."""
        input_json = '{"title": "John said "Hello" to Mary", "summary": "Simple test"}'
        processed = preprocess_json_escapes(input_json)
        # Should be valid JSON after preprocessing
        result = json.loads(processed)
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result
    
    def test_extract_json_object_function(self):
        """Test the extract_json_object function directly."""
        # Test with extra text around JSON object
        text_with_json = 'Here is some text {"title": "Test", "summary": "Content"} and more text'
        extracted = extract_json_object(text_with_json)
        result = json.loads(extracted)
        expected = {"title": "Test", "summary": "Content"}
        assert result == expected
        
        # Test with JSON array
        text_with_array = 'Some text [{"item": "one"}, {"item": "two"}] more text'
        extracted = extract_json_object(text_with_array)
        result = json.loads(extracted)
        expected = [{"item": "one"}, {"item": "two"}]
        assert result == expected
    
    def test_real_world_llm_response_scenarios(self):
        """Test real-world scenarios that might come from LLM responses."""
        
        # Scenario 1: LLM response with explanation text
        llm_response1 = '''Here's the JSON response you requested:

```json
{
  "title": "Q3 Planning Meeting",
  "summary": "We discussed the "new initiative" and John's "breakthrough idea" for next quarter."
}
```

This should help with your transcription needs.'''
        
        result1 = safe_json_loads(llm_response1)
        assert isinstance(result1, dict)
        assert "title" in result1
        assert "summary" in result1
        
        # Scenario 2: LLM response with unescaped quotes and no code blocks
        llm_response2 = '{"title": "Team Standup", "summary": "Alice mentioned "the deadline is tight" and Bob said "we need more resources""}'
        
        result2 = safe_json_loads(llm_response2)
        assert isinstance(result2, dict)
        assert "title" in result2
        assert "summary" in result2
        
        # Scenario 3: LLM response with speaker identification
        llm_response3 = '''{"SPEAKER_00": "John Smith", "SPEAKER_01": "Jane "The Expert" Doe", "SPEAKER_02": "Bob"}'''
        
        result3 = safe_json_loads(llm_response3)
        assert isinstance(result3, dict)
        assert len(result3) >= 2  # Should have parsed at least some speakers
    
    def test_fallback_strategies(self):
        """Test that different parsing strategies work as fallbacks."""
        
        # Test ast.literal_eval fallback for simple cases
        simple_dict = "{'title': 'Simple', 'summary': 'Test'}"
        result = safe_json_loads(simple_dict)
        expected = {"title": "Simple", "summary": "Test"}
        assert result == expected
        
        # Test regex extraction fallback
        messy_response = 'Some text before {"title": "Extracted", "summary": "From regex"} some text after'
        result = safe_json_loads(messy_response)
        expected = {"title": "Extracted", "summary": "From regex"}
        assert result == expected
    
    def test_performance_with_large_content(self):
        """Test performance with larger JSON content."""
        large_summary = "This is a very long summary. " * 100  # Create a long string
        large_json = f'{{"title": "Large Content Test", "summary": "{large_summary}"}}'
        
        result = safe_json_loads(large_json)
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result
        assert result["title"] == "Large Content Test"


class TestComprehensiveJSONScenarios:
    """Comprehensive test scenarios for JSON preprocessing."""
    
    @pytest.mark.parametrize("test_case", [
        {
            "name": "Valid JSON",
            "input": '{"title": "Test", "summary": "Valid JSON"}',
            "should_succeed": True
        },
        {
            "name": "Unescaped quotes in title",
            "input": '{"title": "Meeting about "Project X"", "summary": "Discussion summary"}',
            "should_succeed": True
        },
        {
            "name": "Multiple unescaped quotes",
            "input": '{"title": "John said "Hello" and Mary replied "Hi there"", "summary": "Conversation log"}',
            "should_succeed": True
        },
        {
            "name": "Markdown code block",
            "input": '```json\n{"title": "Wrapped", "summary": "In code block"}\n```',
            "should_succeed": True
        },
        {
            "name": "Mixed quotes and apostrophes",
            "input": '{"title": "Alice\'s "big idea" presentation", "summary": "She said it\'s "revolutionary""}',
            "should_succeed": True
        },
        {
            "name": "JSON with newlines",
            "input": '{"title": "Multi-line", "summary": "Line 1\\nLine 2 with \\"quotes\\"\\nLine 3"}',
            "should_succeed": True
        },
        {
            "name": "Completely malformed",
            "input": '{"title": "Test", "summary": this is not valid json at all}',
            "should_succeed": False
        },
        {
            "name": "Empty string",
            "input": "",
            "should_succeed": False
        }
    ])
    def test_comprehensive_scenarios(self, test_case):
        """Test comprehensive JSON scenarios using parametrization."""
        result = safe_json_loads(test_case['input'], {"error": "fallback"})
        
        if test_case['should_succeed']:
            assert result != {"error": "fallback"}
            assert isinstance(result, (dict, list))
        else:
            assert result == {"error": "fallback"}