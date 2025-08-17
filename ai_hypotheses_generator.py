import json
import numpy as np
from typing import List, Dict, Tuple, Any
import tiktoken
from openai import OpenAI
import re
from pathlib import Path
import time
from dsl import objects, mostcolor

class CategorySpecificAIGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the category-specific AI hypothesis generator"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.hypothesis_cache = {}
        
        # Category-specific prompt templates
        self.category_templates = {
            "Colour": {
                "focus": "color changes, recoloring rules, palette swaps, color-based conditions",
                "examples": "objects changing from blue to red, all pixels of color X become color Y, objects take color of neighboring objects",
                "tests": "color histogram changes, specific color mappings, color-based rules"
            },
            "Movement": {
                "focus": "spatial translation, repositioning, shifting objects in specific directions",
                "examples": "objects move left by 2 pixels, all objects shift down until they hit bottom, pieces slide in cardinal directions",
                "tests": "position changes, directional movement, preservation of object shape during movement"
            },
            "Geometry": {
                "focus": "shape transformations, rotations, reflections, scaling, cropping, resizing",
                "examples": "grid rotated 90 degrees clockwise, objects reflected across vertical axis, shapes scaled up 2x",
                "tests": "rotational symmetry, reflection detection, scaling factors, shape preservation"
            },
            "Object": {
                "focus": "object manipulation, merging, splitting, duplication, creation, deletion",
                "examples": "adjacent objects merge into one, objects split along lines, duplicate objects appear",
                "tests": "object count changes, connectivity changes, object property modifications"
            },
            "Number": {
                "focus": "count-based transformations, repetitions based on numerical properties",
                "examples": "repeat pattern N times where N is number of objects, create X copies based on color count",
                "tests": "numerical relationships, count-based rules, mathematical operations"
            },
            "Commonsense": {
                "focus": "pattern completion, implicit logical reasoning, contextual understanding",
                "examples": "complete missing parts of patterns, apply common sense rules, fill logical gaps",
                "tests": "pattern consistency, logical completion, contextual appropriateness"
            }
        }

    def get_likely_categories(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """Use heuristics to determine which categories are most likely"""
        likely = []
        
        # Quick heuristics
        if input_grid.shape != output_grid.shape:
            likely.append("Geometry")
        
        if set(input_grid.flat) != set(output_grid.flat):
            likely.append("Colour")
        
        input_counts = np.bincount(input_grid.flat, minlength=10)
        output_counts = np.bincount(output_grid.flat, minlength=10)
        if np.array_equal(input_counts, output_counts) and not np.array_equal(input_grid, output_grid):
            likely.append("Movement")
        
        # Check for object-level changes
        try:
            input_objects = objects(tuple(tuple(row) for row in input_grid), True, True, True)
            output_objects = objects(tuple(tuple(row) for row in output_grid), True, True, True)
            if len(input_objects) != len(output_objects):
                likely.append("Object")
        except:
            pass
        
        # Check for numerical patterns
        input_nonzero = np.count_nonzero(input_grid)
        output_nonzero = np.count_nonzero(output_grid)
        if output_nonzero > input_nonzero:
            ratio = output_nonzero / input_nonzero
            if ratio == int(ratio) and ratio > 1:
                likely.append("Number")
        
        # Always include Commonsense as fallback
        if "Commonsense" not in likely:
            likely.append("Commonsense")
        
        # If Object not detected but we have other changes, add it
        if len(likely) < 3 and "Object" not in likely:
            likely.append("Object")
            
        return likely[:3]  # Limit to 3 most likely categories

    def format_grid_for_prompt(self, grid: np.ndarray) -> str:
        """Format grid in a compact, readable way for the prompt"""
        lines = []
        for row in grid:
            lines.append(" ".join(f"{cell:2d}" for cell in row))
        return "\n".join(lines)

    def analyze_transformation_by_category(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                         target_categories: List[str]) -> Dict[str, str]:
        """Analyze the transformation from specific categories' perspectives"""
        input_formatted = self.format_grid_for_prompt(input_grid)
        output_formatted = self.format_grid_for_prompt(output_grid)
        
        # Basic change detection
        size_changed = input_grid.shape != output_grid.shape
        colors_changed = set(input_grid.flat) != set(output_grid.flat)
        positions_changed = not np.array_equal(input_grid, output_grid)
        
        analyses = {}
        
        for category in target_categories:
            if category not in self.category_templates:
                continue
                
            template = self.category_templates[category]
            
            category_prompt = f"""Analyze this transformation SPECIFICALLY from a {category} perspective:

INPUT GRID ({input_grid.shape[0]}x{input_grid.shape[1]}):
{input_formatted}

OUTPUT GRID ({output_grid.shape[0]}x{output_grid.shape[1]}):
{output_formatted}

Focus on: {template['focus']}
Look for patterns like: {template['examples']}

Quick facts:
- Size changed: {size_changed}
- Colors changed: {colors_changed} 
- Positions changed: {positions_changed}

From a {category} perspective:
1. What specific {category.lower()} changes do you observe?
2. Can you describe a precise {category.lower()} rule that explains this transformation?
3. How would you test if this {category.lower()} rule applies to other examples?

Be very specific about {category.lower()} aspects. If this doesn't look like a {category.lower()} transformation, say so clearly."""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"You are an expert in {category.lower()} transformations in visual puzzles. Focus ONLY on {category.lower()} aspects."},
                        {"role": "user", "content": category_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=400
                )
                analyses[category] = response.choices[0].message.content
                
            except Exception as e:
                analyses[category] = f"Analysis failed: {e}"
        
        return analyses

    def generate_category_specific_hypotheses(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                            target_categories: List[str] = None,
                                            num_per_category: int = 2) -> List[Dict[str, Any]]:
        """Generate hypotheses specifically designed for target categories"""
        
        # Create cache key
        cache_key = f"{hash(input_grid.tobytes())}_{hash(output_grid.tobytes())}_{'_'.join(target_categories or [])}"
        if cache_key in self.hypothesis_cache:
            return self.hypothesis_cache[cache_key]
        
        # Use heuristics to determine likely categories if not specified
        if target_categories is None:
            target_categories = self.get_likely_categories(input_grid, output_grid)
        
        print(f"🎯 Focusing on categories: {target_categories}")
        
        # Get category-specific analyses for target categories only
        category_analyses = self.analyze_transformation_by_category(input_grid, output_grid, target_categories)
        
        all_hypotheses = []
        
        for category, analysis in category_analyses.items():
            template = self.category_templates[category]
            
            hypothesis_prompt = f"""Based on this {category} analysis:

{analysis}

Generate {num_per_category} HIGHLY SPECIFIC {category} hypotheses for this transformation.

Requirements for {category} hypotheses:
- Focus EXCLUSIVELY on {template['focus']}
- Test for {template['tests']}
- Be precise about {category.lower()} mechanisms
- Each hypothesis should be clearly testable as a {category} transformation
- If this is NOT a {category} transformation, create hypotheses that would FAIL the tests

Return as JSON array:
[
  {{
    "description": "Precise {category.lower()}-focused description of the transformation rule",
    "category": "{category}",
    "confidence": 0.8,
    "evidence": "Specific {category.lower()} evidence from the grids",
    "prior_probability": 0.15,
    "specificity_score": 0.9
  }}
]

Make each hypothesis DISTINCTLY about {category.lower()} aspects."""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"Generate ONLY {category} hypotheses. Be extremely specific about {category.lower()} transformations. Return valid JSON."},
                        {"role": "user", "content": hypothesis_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    hypotheses = json.loads(json_match.group())
                    
                    # Generate category-specific test code
                    for hyp in hypotheses:
                        hyp['test_code'] = self.generate_category_specific_test_code(
                            hyp['description'], category, input_grid, output_grid
                        )
                        hyp['source'] = f'category_specific_{category.lower()}'
                    
                    all_hypotheses.extend(hypotheses)
                    
            except Exception as e:
                print(f"Error generating {category} hypotheses: {e}")
                # Add fallback hypothesis for this category
                fallback = self.create_category_fallback_hypothesis(category, input_grid, output_grid)
                all_hypotheses.append(fallback)
        
        # Cache and return
        self.hypothesis_cache[cache_key] = all_hypotheses
        return all_hypotheses

    def generate_category_specific_test_code(self, description: str, category: str, 
                                           input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Generate highly specific test code for each category with improved logic"""
        
        if category == "Colour":
            return f"""
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if input_grid.shape != output_grid.shape:
            return False
        
        # Get color sets
        input_colors = set(input_grid.flat)
        output_colors = set(output_grid.flat)
        
        # No change at all - not a color transformation
        if input_colors == output_colors and np.array_equal(input_grid, output_grid):
            return False
            
        # Check for position-preserving color changes
        color_mapping = {{}}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]
                if in_color in color_mapping:
                    if color_mapping[in_color] != out_color:
                        return False  # Inconsistent mapping
                else:
                    color_mapping[in_color] = out_color
        
        # At least one color must actually change
        return any(k != v for k, v in color_mapping.items())
    except:
        return False
"""
        
        elif category == "Movement":
            return f"""
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if input_grid.shape != output_grid.shape:
            return False
        
        # Check if colors are preserved (movement doesn't change colors)
        input_counts = np.bincount(input_grid.flat, minlength=10)
        output_counts = np.bincount(output_grid.flat, minlength=10)
        
        if not np.array_equal(input_counts, output_counts):
            return False  # Colors changed, not pure movement
        
        if np.array_equal(input_grid, output_grid):
            return False  # No movement occurred
        
        # Check if this looks like a movement pattern
        # (pixels moved but overall structure preserved)
        non_zero_input = np.count_nonzero(input_grid)
        non_zero_output = np.count_nonzero(output_grid)
        
        # Must preserve non-zero pixel count for movement
        if non_zero_input != non_zero_output:
            return False
            
        # Additional check: see if we can find a simple translation
        # that explains the transformation
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dy == 0 and dx == 0:
                    continue
                # Test if shifting by (dy, dx) produces the output
                shifted = np.zeros_like(input_grid)
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        new_i, new_j = i + dy, j + dx
                        if 0 <= new_i < shifted.shape[0] and 0 <= new_j < shifted.shape[1]:
                            shifted[new_i, new_j] = input_grid[i, j]
                
                if np.array_equal(shifted, output_grid):
                    return True
        
        return True  # Default to movement if colors preserved and positions changed
    except:
        return False
"""
        
        elif category == "Geometry":
            return f"""
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        
        # Size change is geometric
        if input_grid.shape != output_grid.shape:
            return True
        
        # Check standard geometric transformations
        # Rotations
        for k in range(1, 4):
            if np.array_equal(output_grid, np.rot90(input_grid, k)):
                return True
        
        # Reflections
        if (np.array_equal(output_grid, np.flip(input_grid, axis=0)) or
            np.array_equal(output_grid, np.flip(input_grid, axis=1))):
            return True
        
        # Transpose
        if np.array_equal(output_grid, input_grid.T):
            return True
        
        # Check if colors are preserved (geometric transforms preserve colors)
        input_counts = np.bincount(input_grid.flat, minlength=10)
        output_counts = np.bincount(output_grid.flat, minlength=10)
        
        # If colors preserved but positions changed, might be geometric
        if np.array_equal(input_counts, output_counts) and not np.array_equal(input_grid, output_grid):
            return True
            
        return False
    except:
        return False
"""
        
        elif category == "Object":
            return f"""
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if input_objects is None or output_objects is None:
            # Fallback: check for changes that might indicate object manipulation
            if not np.array_equal(input_grid, output_grid):
                return True
            return False
        
        # Check for object-level changes
        input_count = len(input_objects)
        output_count = len(output_objects)
        
        # Object count changed - this is object manipulation
        if input_count != output_count:
            return True
        
        # Check if object properties changed
        for i in range(min(len(input_objects), len(output_objects))):
            in_obj = input_objects[i]
            out_obj = output_objects[i]
            
            # Object size changed
            if len(in_obj) != len(out_obj):
                return True
            
            # Check colors within objects
            in_colors = set(color for color, pos in in_obj) if hasattr(in_obj[0], '__iter__') and len(in_obj[0]) == 2 else set()
            out_colors = set(color for color, pos in out_obj) if hasattr(out_obj[0], '__iter__') and len(out_obj[0]) == 2 else set()
            if in_colors != out_colors:
                return True
        
        return False  # No significant object changes detected
    except:
        # Fallback test
        return not np.array_equal(input_grid, output_grid)
"""
        
        elif category == "Number":
            return f"""
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        # Look for numerical patterns and relationships
        
        input_nonzero = np.count_nonzero(input_grid)
        output_nonzero = np.count_nonzero(output_grid)
        
        # Check for multiplication/repetition patterns
        if output_nonzero > input_nonzero:
            ratio = output_nonzero / input_nonzero
            if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                return True
        
        # Check if transformation involves counting
        if input_objects and output_objects:
            input_obj_count = len(input_objects)
            output_obj_count = len(output_objects)
            
            # Numerical relationship between object counts
            if output_obj_count > input_obj_count:
                ratio = output_obj_count / input_obj_count
                if abs(ratio - round(ratio)) < 0.1 and ratio > 1:
                    return True
        
        # Check for grid size relationships
        input_size = input_grid.shape[0] * input_grid.shape[1]
        output_size = output_grid.shape[0] * output_grid.shape[1]
        
        if output_size != input_size:
            ratio = output_size / input_size
            if abs(ratio - round(ratio)) < 0.1:
                return True
        
        # Check for repeating patterns
        if input_grid.shape == output_grid.shape:
            # See if output contains multiple copies of input pattern
            h, w = input_grid.shape
            for scale in [2, 3, 4]:
                if output_grid.shape[0] % scale == 0 and output_grid.shape[1] % scale == 0:
                    sub_h, sub_w = h // scale, w // scale
                    if sub_h > 0 and sub_w > 0:
                        # Check if pattern repeats
                        pattern = input_grid[:sub_h, :sub_w]
                        matches = 0
                        for i in range(0, h, sub_h):
                            for j in range(0, w, sub_w):
                                if np.array_equal(output_grid[i:i+sub_h, j:j+sub_w], pattern):
                                    matches += 1
                        if matches >= scale:
                            return True
        
        return False
    except:
        return False
"""
        
        else:  # Commonsense
            return f"""
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        # Test for pattern completion or logical reasoning
        
        if np.array_equal(input_grid, output_grid):
            return False  # No transformation
        
        # Check if this transformation doesn't fit other categories
        # (If it's not clearly color, movement, geometry, object, or number)
        
        # Check if it's NOT a simple geometric transformation
        is_geometric = False
        if input_grid.shape == output_grid.shape:
            for k in range(1, 4):
                if np.array_equal(output_grid, np.rot90(input_grid, k)):
                    is_geometric = True
                    break
            if not is_geometric:
                for flip_axis in [0, 1]:
                    if np.array_equal(output_grid, np.flip(input_grid, axis=flip_axis)):
                        is_geometric = True
                        break
        
        # Check if it's NOT a simple color mapping
        is_simple_color = False
        if input_grid.shape == output_grid.shape:
            input_colors = set(input_grid.flat)
            output_colors = set(output_grid.flat)
            
            # Test for consistent color mapping
            color_mapping = {{}}
            consistent_mapping = True
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    in_color = input_grid[i, j]
                    out_color = output_grid[i, j]
                    if in_color in color_mapping:
                        if color_mapping[in_color] != out_color:
                            consistent_mapping = False
                            break
                    else:
                        color_mapping[in_color] = out_color
                if not consistent_mapping:
                    break
            
            is_simple_color = consistent_mapping and any(k != v for k, v in color_mapping.items())
        
        # Check if it's NOT simple movement (color counts preserved)
        is_simple_movement = False
        if input_grid.shape == output_grid.shape:
            input_counts = np.bincount(input_grid.flat, minlength=10)
            output_counts = np.bincount(output_grid.flat, minlength=10)
            is_simple_movement = np.array_equal(input_counts, output_counts)
        
        # Commonsense transformations are often complex, context-dependent
        # Return True if it doesn't fit simple patterns
        return not (is_geometric or is_simple_color or is_simple_movement)
    except:
        return True  # Default to commonsense for complex cases
"""

    def create_category_fallback_hypothesis(self, category: str, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Create a fallback hypothesis for a specific category"""
        
        fallback_descriptions = {
            "Colour": "Objects change color according to a systematic color mapping rule",
            "Movement": "Objects translate spatially while preserving their shape and color",
            "Geometry": "The grid undergoes a geometric transformation like rotation or reflection",
            "Object": "Objects in the scene are modified, merged, split, or duplicated",
            "Number": "The transformation follows a numerical pattern based on counts or repetitions",
            "Commonsense": "The transformation completes a logical pattern or applies contextual reasoning"
        }
        
        return {
            "description": fallback_descriptions[category],
            "category": category,
            "confidence": 0.3,
            "evidence": f"Fallback hypothesis for {category} category",
            "prior_probability": 0.1,
            "source": f"fallback_{category.lower()}",
            "test_code": self.generate_category_specific_test_code(
                fallback_descriptions[category], category, input_grid, output_grid
            )
        }

    def rule_complexity(self, rule_description: str) -> int:
        """Calculate complexity based on token count"""
        tokens = self.tokenizer.encode(rule_description)
        return max(len(tokens), 1)

# Integration wrapper to replace your existing AIHypothesisGenerator
class AIHypothesisGenerator(CategorySpecificAIGenerator):
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
    
    def generate_smart_hypotheses(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                num_hypotheses: int = 6) -> List[Dict[str, Any]]:
        """Generate category-specific hypotheses using heuristics for efficiency"""
        
        # Use heuristics to focus on likely categories
        likely_categories = self.get_likely_categories(input_grid, output_grid)
        
        # Calculate how many per category (ensuring we get diverse hypotheses)
        base_per_category = max(1, num_hypotheses // len(likely_categories))
        
        hypotheses = self.generate_category_specific_hypotheses(
            input_grid, output_grid, 
            target_categories=likely_categories,
            num_per_category=base_per_category
        )
        
        # If we have too many, select the best ones from each category
        if len(hypotheses) > num_hypotheses:
            # Group by category and select top ones from each
            by_category = {}
            for hyp in hypotheses:
                cat = hyp['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(hyp)
            
            # Select best from each category
            selected = []
            for cat_hyps in by_category.values():
                # Sort by confidence and take top ones
                cat_hyps.sort(key=lambda x: -x.get('confidence', 0))
                selected.extend(cat_hyps[:base_per_category])
            
            hypotheses = selected[:num_hypotheses]
        
        return hypotheses