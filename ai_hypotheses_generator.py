import json
import numpy as np
from typing import List, Dict, Tuple, Any
import tiktoken
from openai import OpenAI
import re
from pathlib import Path
import time

class AIHypothesisGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the optimized AI hypothesis generator"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Cache for generated hypotheses to avoid re-generation
        self.hypothesis_cache = {}
        self.pattern_cache = {}
        
        # Pre-defined hypothesis templates for common patterns (fast fallback)
        self.template_hypotheses = {
            "color_change": {
                "description": "All pixels of color X change to color Y",
                "category": "Colour",
                "test_code": """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    if input_grid.shape != output_grid.shape:
        return False
    unique_in = set(input_grid.flat)
    unique_out = set(output_grid.flat)
    # Check if it's a simple color mapping
    if len(unique_in) == len(unique_out):
        mapping = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]
                if in_color in mapping and mapping[in_color] != out_color:
                    return False
                mapping[in_color] = out_color
        return len(mapping) > 0
    return False
""",
                "prior_probability": 0.15,
                "confidence": 0.8
            },
            "object_movement": {
                "description": "Objects move by a fixed offset",
                "category": "Movement", 
                "test_code": """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    if input_grid.shape != output_grid.shape:
        return False
    # Simple check: non-zero pixels shift consistently
    in_nonzero = np.argwhere(input_grid != 0)
    out_nonzero = np.argwhere(output_grid != 0)
    if len(in_nonzero) != len(out_nonzero):
        return False
    if len(in_nonzero) == 0:
        return False
    # Check if there's a consistent offset
    if len(in_nonzero) > 1:
        offset = out_nonzero[0] - in_nonzero[0]
        for i in range(len(in_nonzero)):
            expected = in_nonzero[i] + offset
            if not any(np.array_equal(expected, out_pos) for out_pos in out_nonzero):
                return False
    return True
""",
                "prior_probability": 0.12,
                "confidence": 0.7
            },
            "size_change": {
                "description": "Grid size changes (cropping or padding)",
                "category": "Geometry",
                "test_code": """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    return input_grid.shape != output_grid.shape
""",
                "prior_probability": 0.10,
                "confidence": 0.9
            },
            "pattern_fill": {
                "description": "Empty spaces filled with pattern or color",
                "category": "Commonsense",
                "test_code": """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    if input_grid.shape != output_grid.shape:
        return False
    # Check if output has fewer zeros (background)
    in_zeros = np.sum(input_grid == 0)
    out_zeros = np.sum(output_grid == 0)
    return out_zeros < in_zeros and in_zeros > 0
""",
                "prior_probability": 0.11,
                "confidence": 0.6
            },
            "object_count": {
                "description": "Transformation based on counting objects",
                "category": "Number",
                "test_code": """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    if input_objects is None or output_objects is None:
        return False
    # Simple check: output relates to input object count
    in_count = len(input_objects)
    out_count = len(output_objects)
    return in_count != out_count and in_count > 0
""",
                "prior_probability": 0.08,
                "confidence": 0.5
            },
            "object_merge": {
                "description": "Objects are merged, split, or transformed",
                "category": "Object",
                "test_code": """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    if input_objects is None or output_objects is None:
        return False
    # Check if objects changed
    in_count = len(input_objects) 
    out_count = len(output_objects)
    if in_count == out_count == 0:
        return False
    # Objects exist and counts differ, or positions changed
    return in_count != out_count or in_count > 0
""",
                "prior_probability": 0.09,
                "confidence": 0.6
            }
        }

    def quick_pattern_analysis(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Quick pattern analysis to determine if we need AI or can use templates"""
        # Create a simple pattern signature
        in_shape = input_grid.shape
        out_shape = output_grid.shape
        in_colors = len(np.unique(input_grid))
        out_colors = len(np.unique(output_grid))
        
        pattern_key = f"{in_shape}→{out_shape}_{in_colors}→{out_colors}"
        
        # Check cache first
        if pattern_key in self.pattern_cache:
            return self.pattern_cache[pattern_key]
        
        # Quick heuristics
        if in_shape != out_shape:
            self.pattern_cache[pattern_key] = "geometry"
        elif in_colors != out_colors:
            self.pattern_cache[pattern_key] = "color" 
        elif np.array_equal(input_grid, output_grid):
            self.pattern_cache[pattern_key] = "identity"
        else:
            self.pattern_cache[pattern_key] = "complex"
            
        return self.pattern_cache[pattern_key]

    def generate_hypotheses_optimized(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                    use_ai: bool = True) -> List[Dict[str, Any]]:
        """Optimized hypothesis generation with smart caching and templates"""
        
        # Create cache key
        cache_key = f"{hash(input_grid.tobytes())}_{hash(output_grid.tobytes())}"
        if cache_key in self.hypothesis_cache:
            return self.hypothesis_cache[cache_key]
        
        pattern_type = self.quick_pattern_analysis(input_grid, output_grid)
        
        hypotheses = []
        
        # Always include fast template-based hypotheses first
        for template_name, template in self.template_hypotheses.items():
            hypotheses.append({
                **template,
                "source": "template",
                "template_name": template_name
            })
        
        # Only use expensive AI if pattern is complex and we need it
        if use_ai and pattern_type == "complex" and len(hypotheses) < 3:
            try:
                ai_hypotheses = self.generate_ai_hypotheses_batch(input_grid, output_grid, max_hypotheses=3)
                hypotheses.extend(ai_hypotheses)
            except Exception as e:
                print(f"AI generation failed: {e}, using templates only")
        
        # Cache result
        self.hypothesis_cache[cache_key] = hypotheses
        return hypotheses

    def generate_ai_hypotheses_batch(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                   max_hypotheses: int = 3) -> List[Dict[str, Any]]:
        """Generate AI hypotheses with optimized prompting"""
        
        # Minimal grid description to save tokens
        input_desc = self.minimal_grid_description(input_grid)
        output_desc = self.minimal_grid_description(output_grid)
        
        # Shorter, focused prompt
        prompt = f"""Analyze this ARC transformation (be concise):

INPUT: {input_desc}
OUTPUT: {output_desc}

Generate {max_hypotheses} hypotheses as JSON:
[{{"description": "brief rule", "category": "Colour|Commonsense|Geometry|Movement|Number|Object", "confidence": 0.8, "prior_probability": 0.1}}]

Categories:
- Colour: color changes
- Movement: position changes  
- Geometry: shape/size changes
- Number: count-based rules
- Object: object manipulation
- Commonsense: pattern completion

Focus on the most obvious transformation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cheaper than GPT-4
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=300    # Limit tokens
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                ai_hypotheses = json.loads(json_match.group())
                
                # Add test code generation for AI hypotheses
                for hyp in ai_hypotheses:
                    hyp['test_code'] = self.generate_simple_test_code(hyp)
                    hyp['source'] = 'ai'
                
                return ai_hypotheses
            else:
                return []
                
        except Exception as e:
            print(f"Error in AI generation: {e}")
            return []

    def minimal_grid_description(self, grid: np.ndarray) -> str:
        """Minimal description to save prompt tokens"""
        h, w = grid.shape
        colors = np.unique(grid)
        return f"{h}x{w}, colors:{list(colors)}"

    def generate_simple_test_code(self, hypothesis: Dict[str, Any]) -> str:
        """Generate simple test code based on category and description"""
        category = hypothesis['category']
        
        # Template-based code generation (faster than AI)
        if category == "Colour":
            return """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    return not np.array_equal(input_grid, output_grid) and input_grid.shape == output_grid.shape
"""
        elif category == "Movement":
            return """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    return (input_grid.shape == output_grid.shape and 
            np.sum(input_grid) == np.sum(output_grid) and 
            not np.array_equal(input_grid, output_grid))
"""
        elif category == "Geometry":
            return """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    return input_grid.shape != output_grid.shape
"""
        else:
            return """
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    return not np.array_equal(input_grid, output_grid)
"""

    def rule_complexity(self, rule_description: str) -> int:
        """Calculate complexity based on token count"""
        tokens = self.tokenizer.encode(rule_description)
        return max(len(tokens), 1)  # Avoid zero complexity