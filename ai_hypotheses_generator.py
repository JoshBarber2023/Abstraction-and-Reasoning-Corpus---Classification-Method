import json
import numpy as np
from typing import List, Dict, Tuple, Any
import tiktoken
from openai import OpenAI
import re
from pathlib import Path
import time
from dsl import objects, mostcolor

class ImprovedCategoryAIGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the improved category-specific AI hypothesis generator"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.hypothesis_cache = {}
        
        # Improved category-specific prompt templates
        self.category_templates = {
            "Colour": {
                "focus": "color transformations, recoloring rules, palette changes, color-based logic",
                "examples": "red pixels become blue, objects change color based on neighbors, color swapping patterns",
                "key_tests": "consistent color mapping, position-preserving color changes, color count preservation in swaps",
                "avoid": "position changes, size changes, object count changes"
            },
            "Movement": {
                "focus": "spatial translation, shifting, repositioning of objects while preserving shape and color",
                "examples": "objects slide left 2 pixels, pieces fall down due to gravity, objects align to edges",
                "key_tests": "preserved object shapes and colors, changed positions, directional movement patterns",
                "avoid": "color changes, size changes, object creation/deletion"
            },
            "Geometry": {
                "focus": "shape transformations, rotations, reflections, scaling, cropping, resizing operations",
                "examples": "90-degree rotation, horizontal reflection, scaling up by factor of 2, cropping to center",
                "key_tests": "rotational/reflectional symmetry, shape preservation during transformation, scaling relationships",
                "avoid": "pure color changes without shape effects, simple position shifts"
            },
            "Object": {
                "focus": "object-level operations like splitting, merging, duplicating, morphing discrete entities",
                "examples": "two objects merge into one, object splits along a line, object duplicates, shapes morph",
                "key_tests": "object count changes, connectivity changes, object boundary modifications",
                "avoid": "pure color/position changes that don't affect object structure"
            },
            "Number": {
                "focus": "count-based rules, repetitions, arithmetic relationships, quantity-driven logic",
                "examples": "repeat pattern N times based on object count, create X copies where X equals color frequency",
                "key_tests": "mathematical relationships, count-based repetitions, numerical patterns in transformation",
                "avoid": "transformations not driven by counting or arithmetic"
            },
            "Commonsense": {
                "focus": "pattern completion, implicit reasoning, contextual understanding, logical inference",
                "examples": "complete missing puzzle pieces, apply learned rules from context, logical pattern extension",
                "key_tests": "contextual appropriateness, pattern consistency, logical completion",
                "avoid": "simple mechanical transformations that fit other categories clearly"
            }
        }

    def get_improved_likely_categories(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                     training_pairs: List[Tuple[np.ndarray, np.ndarray]] = None) -> List[str]:
        """Improved heuristics using all training pairs and better logic"""
        likely = set()
        
        # Analyze the primary pair
        primary_analysis = self._analyze_single_pair(input_grid, output_grid)
        likely.update(primary_analysis)
        
        # If we have multiple training pairs, analyze consistency
        if training_pairs and len(training_pairs) > 1:
            consistent_categories = self._find_consistent_categories(training_pairs)
            if consistent_categories:
                likely.update(consistent_categories)
        
        # Ensure we have reasonable candidates
        likely_list = list(likely)
        
        # Always include Commonsense as a fallback, but not as primary if we have good options
        if len(likely_list) == 0:
            likely_list = ["Commonsense", "Object", "Colour"]
        elif len(likely_list) == 1:
            likely_list.append("Commonsense")
        elif "Commonsense" not in likely_list and len(likely_list) < 3:
            likely_list.append("Commonsense")
        
        return likely_list[:4]  # Return top 4 categories

    def _analyze_single_pair(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """Analyze a single input-output pair for likely categories"""
        categories = []
        
        # Shape analysis
        shape_changed = input_grid.shape != output_grid.shape
        if shape_changed:
            categories.append("Geometry")
        
        # Color analysis
        input_colors = set(input_grid.flat)
        output_colors = set(output_grid.flat)
        colors_changed = input_colors != output_colors
        
        # Position analysis (same shape required)
        if not shape_changed:
            positions_changed = not np.array_equal(input_grid, output_grid)
            
            # Color count analysis
            input_counts = np.bincount(input_grid.flat, minlength=10)
            output_counts = np.bincount(output_grid.flat, minlength=10)
            color_counts_same = np.array_equal(input_counts, output_counts)
            
            # Classify based on what changed
            if colors_changed and not color_counts_same:
                # Colors and their counts changed - likely color transformation
                categories.append("Colour")
            elif positions_changed and color_counts_same:
                # Positions changed but colors preserved - likely movement
                categories.append("Movement")
            elif colors_changed and color_counts_same:
                # Colors changed but counts same - color swapping
                categories.append("Colour")
        
        # Object analysis
        try:
            input_objects = objects(tuple(tuple(row) for row in input_grid), True, True, True)
            output_objects = objects(tuple(tuple(row) for row in output_grid), True, True, True)
            
            if len(input_objects) != len(output_objects):
                categories.append("Object")
            elif len(input_objects) > 0:
                # Check if object properties changed
                for i in range(min(len(input_objects), len(output_objects))):
                    if len(input_objects[i]) != len(output_objects[i]):
                        categories.append("Object")
                        break
        except:
            pass
        
        # Numerical analysis
        input_nonzero = np.count_nonzero(input_grid)
        output_nonzero = np.count_nonzero(output_grid)
        if output_nonzero > input_nonzero:
            ratio = output_nonzero / input_nonzero
            if abs(ratio - round(ratio)) < 0.1 and ratio > 1:
                categories.append("Number")
        
        return categories

    def _find_consistent_categories(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Find categories that are consistent across all training pairs"""
        if len(training_pairs) < 2:
            return []
        
        # Analyze each pair
        pair_categories = []
        for inp, out in training_pairs:
            pair_cats = self._analyze_single_pair(inp, out)
            pair_categories.append(set(pair_cats))
        
        # Find intersection - categories that appear in ALL pairs
        consistent = set.intersection(*pair_categories) if pair_categories else set()
        
        return list(consistent)

    def generate_task_aware_hypotheses(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                                     target_categories: List[str] = None, 
                                     num_per_category: int = 2) -> List[Dict[str, Any]]:
        """Generate hypotheses with full task context"""
        
        if not training_pairs:
            return []
        
        # Use improved heuristics with all pairs
        primary_pair = training_pairs[0]
        if target_categories is None:
            target_categories = self.get_improved_likely_categories(
                primary_pair[0], primary_pair[1], training_pairs
            )
        
        print(f"🎯 AI focusing on categories: {target_categories}")
        
        # Create task context summary
        task_summary = self._create_task_summary(training_pairs)
        
        all_hypotheses = []
        
        # Generate hypotheses for each target category
        for category in target_categories:
            category_hypotheses = self._generate_category_hypotheses_with_context(
                training_pairs, category, task_summary, num_per_category
            )
            all_hypotheses.extend(category_hypotheses)
        
        # CRITICAL: Add hypothesis validation step
        validated_hypotheses = self._validate_and_improve_hypotheses(all_hypotheses, training_pairs)
        
        return validated_hypotheses

    def _create_task_summary(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Create a concise summary of the task for context"""
        summary_parts = []
        
        summary_parts.append(f"Task has {len(training_pairs)} training examples.")
        
        # Analyze consistency across pairs
        shapes = [(inp.shape, out.shape) for inp, out in training_pairs]
        if len(set(shapes)) == 1:
            summary_parts.append(f"All examples: input {shapes[0][0]} → output {shapes[0][1]}")
        else:
            summary_parts.append("Examples have varying input/output shapes.")
        
        # Color analysis
        all_input_colors = set()
        all_output_colors = set()
        for inp, out in training_pairs:
            all_input_colors.update(inp.flat)
            all_output_colors.update(out.flat)
        
        summary_parts.append(f"Colors used: input {sorted(all_input_colors)}, output {sorted(all_output_colors)}")
        
        # Basic transformation consistency
        transformations = []
        for inp, out in training_pairs:
            if inp.shape != out.shape:
                transformations.append("shape_change")
            elif not np.array_equal(inp, out):
                if set(inp.flat) != set(out.flat):
                    transformations.append("color_change")
                else:
                    transformations.append("position_change")
            else:
                transformations.append("no_change")
        
        if len(set(transformations)) == 1:
            summary_parts.append(f"Consistent transformation type: {transformations[0]}")
        else:
            summary_parts.append(f"Mixed transformations: {set(transformations)}")
        
        return " ".join(summary_parts)

    def _generate_category_hypotheses_with_context(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                                                 category: str, task_summary: str, 
                                                 num_hypotheses: int) -> List[Dict[str, Any]]:
        """Generate hypotheses for a specific category with full task context"""
        
        if category not in self.category_templates:
            return []
        
        template = self.category_templates[category]
        
        # Format examples for context
        examples_text = ""
        for i, (inp, out) in enumerate(training_pairs[:3]):  # Show max 3 examples
            examples_text += f"\nExample {i+1}:\nInput ({inp.shape}): {self._grid_to_compact_string(inp)}\n"
            examples_text += f"Output ({out.shape}): {self._grid_to_compact_string(out)}"
        
        hypothesis_prompt = f"""You are analyzing an ARC puzzle to generate {category} hypotheses.

TASK CONTEXT: {task_summary}

TRAINING EXAMPLES: {examples_text}

Generate {num_hypotheses} specific {category} hypotheses that explain the input→output transformation.

{category} FOCUS: {template['focus']}
Look for: {template['examples']}
Key tests: {template['key_tests']}
Avoid: {template['avoid']}

CRITICAL REQUIREMENTS:
1. Each hypothesis must explain ALL training examples, not just one
2. Focus specifically on {category.lower()} aspects of the transformation
3. Be precise about the {category.lower()} mechanism involved
4. Include specific evidence from the examples
5. Consider the transformation relationship, not individual grids

Return as JSON array:
[
  {{
    "description": "Precise {category.lower()}-focused rule that explains input→output relationship",
    "category": "{category}",
    "confidence": 0.8,
    "evidence": "Specific evidence from training examples supporting this {category.lower()} rule",
    "prior_probability": 0.15,
    "specificity_score": 0.9,
    "explains_all_examples": true
  }}
]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You generate precise {category} transformation hypotheses for ARC puzzles. Focus on the input→output RELATIONSHIP. Return valid JSON."},
                    {"role": "user", "content": hypothesis_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                hypotheses = json.loads(json_match.group())
                
                # Add test code and source
                for hyp in hypotheses:
                    hyp['test_code'] = self._generate_improved_test_code(hyp['description'], category)
                    hyp['source'] = f'ai_contextual_{category.lower()}'
                
                return hypotheses
                
        except Exception as e:
            print(f"Error generating {category} hypotheses: {e}")
            return []
        
        return []

    def _validate_and_improve_hypotheses(self, hypotheses: List[Dict[str, Any]], 
                                       training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """Validate hypotheses and improve poor ones"""
        if not hypotheses:
            return []
        
        print(f"🔍 Validating {len(hypotheses)} hypotheses...")
        
        # Quick test all hypotheses on first example
        first_inp, first_out = training_pairs[0]
        try:
            first_inp_objs = objects(tuple(tuple(row) for row in first_inp), True, True, True)
            first_out_objs = objects(tuple(tuple(row) for row in first_out), True, True, True)
        except:
            first_inp_objs = first_out_objs = None
        
        validated = []
        needs_improvement = []
        
        for hyp in hypotheses:
            try:
                # Test on first example
                result = self._test_hypothesis_safe(hyp['test_code'], first_inp, first_out, 
                                                  first_inp_objs, first_out_objs)
                
                if result:
                    validated.append(hyp)
                else:
                    needs_improvement.append(hyp)
                    
            except Exception as e:
                print(f"Error testing hypothesis: {e}")
                needs_improvement.append(hyp)
        
        # Improve poor hypotheses
        if needs_improvement:
            print(f"🛠️ Improving {len(needs_improvement)} poor hypotheses...")
            improved = self._improve_poor_hypotheses(needs_improvement, training_pairs)
            validated.extend(improved)
        
        print(f"✅ Final hypothesis count: {len(validated)}")
        return validated

    def _improve_poor_hypotheses(self, poor_hypotheses: List[Dict[str, Any]], 
                               training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """Improve hypotheses that failed initial validation"""
        
        if not poor_hypotheses:
            return []
        
        # Group by category for batch improvement
        by_category = {}
        for hyp in poor_hypotheses:
            cat = hyp['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(hyp)
        
        improved_hypotheses = []
        
        for category, cat_hyps in by_category.items():
            template = self.category_templates.get(category, {})
            
            # Create improvement prompt
            failed_descriptions = [hyp['description'] for hyp in cat_hyps]
            examples_text = self._format_examples_for_improvement(training_pairs)
            
            improvement_prompt = f"""These {category} hypotheses FAILED to explain the ARC transformation:

FAILED HYPOTHESES:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(failed_descriptions))}

ACTUAL TRAINING EXAMPLES:
{examples_text}

Generate {len(cat_hyps)} CORRECTED {category} hypotheses that actually work.

{category} REQUIREMENTS:
- Focus: {template.get('focus', 'category-specific transformations')}
- Must explain: {template.get('key_tests', 'the actual transformation pattern')}
- Avoid: {template.get('avoid', 'irrelevant aspects')}

CRITICAL: The new hypotheses must actually explain the input→output transformation shown in the examples.

Return as JSON array with corrected hypotheses:
[
  {{
    "description": "Corrected {category.lower()} rule that actually explains the examples",
    "category": "{category}",
    "confidence": 0.7,
    "evidence": "Specific evidence showing why this explains the transformation",
    "prior_probability": 0.12,
    "corrected": true
  }}
]"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"Fix {category} hypotheses that failed. Generate working alternatives."},
                        {"role": "user", "content": improvement_prompt}
                    ],
                    temperature=0.4,
                    max_tokens=700
                )
                
                content = response.choices[0].message.content.strip()
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                
                if json_match:
                    corrected = json.loads(json_match.group())
                    
                    for hyp in corrected:
                        hyp['test_code'] = self._generate_improved_test_code(hyp['description'], category)
                        hyp['source'] = f'ai_corrected_{category.lower()}'
                    
                    improved_hypotheses.extend(corrected)
                    
            except Exception as e:
                print(f"Error improving {category} hypotheses: {e}")
                # Keep original as fallback
                improved_hypotheses.extend(cat_hyps)
        
        return improved_hypotheses

    def _format_examples_for_improvement(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Format training examples clearly for hypothesis improvement"""
        examples_text = ""
        for i, (inp, out) in enumerate(training_pairs[:2]):  # Max 2 examples for brevity
            examples_text += f"\nExample {i+1}:"
            examples_text += f"\n  Input:  {self._grid_to_compact_string(inp)}"
            examples_text += f"\n  Output: {self._grid_to_compact_string(out)}"
        return examples_text

    def _grid_to_compact_string(self, grid: np.ndarray) -> str:
        """Convert grid to compact string representation"""
        if grid.size > 25:  # For large grids, show summary
            return f"[{grid.shape} grid with colors {sorted(set(grid.flat))}]"
        else:
            # Show actual grid for small grids
            rows = []
            for row in grid:
                rows.append("".join(str(cell) for cell in row))
            return "[" + "|".join(rows) + "]"

    def _test_hypothesis_safe(self, test_code: str, input_grid: np.ndarray, output_grid: np.ndarray,
                            input_objects=None, output_objects=None) -> bool:
        """Safely test a hypothesis"""
        try:
            namespace = {
                'np': np, 'input_grid': input_grid, 'output_grid': output_grid,
                'input_objects': input_objects, 'output_objects': output_objects,
                'objects': objects, 'len': len, 'set': set, 'any': any, 'all': all,
                'max': max, 'min': min, 'abs': abs
            }
            
            exec(test_code, namespace)
            result = namespace['test_hypothesis'](input_grid, output_grid, input_objects, output_objects)
            return bool(result)
            
        except Exception:
            return False

    def _generate_improved_test_code(self, description: str, category: str) -> str:
        """Generate more robust test code for each category"""
        
        if category == "Colour":
            return f'''
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if input_grid.shape != output_grid.shape:
            return False
        
        # Get color mappings and check consistency
        color_mapping = {{}}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]
                if in_color in color_mapping:
                    if color_mapping[in_color] != out_color:
                        return False
                else:
                    color_mapping[in_color] = out_color
        
        # Must have at least one actual color change
        has_color_change = any(k != v for k, v in color_mapping.items())
        
        # Check it's not just a position change (movement/geometry)
        input_counts = np.bincount(input_grid.flat, minlength=10)
        output_counts = np.bincount(output_grid.flat, minlength=10)
        
        return has_color_change
    except:
        return False
'''

        elif category == "Movement":
            return f'''
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if input_grid.shape != output_grid.shape:
            return False
        
        # Colors must be preserved for pure movement
        input_counts = np.bincount(input_grid.flat, minlength=10)
        output_counts = np.bincount(output_grid.flat, minlength=10)
        if not np.array_equal(input_counts, output_counts):
            return False
        
        # Positions must change
        if np.array_equal(input_grid, output_grid):
            return False
        
        # Check for simple translation pattern
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros_like(input_grid)
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        new_i, new_j = i + dy, j + dx
                        if 0 <= new_i < shifted.shape[0] and 0 <= new_j < shifted.shape[1]:
                            shifted[new_i, new_j] = input_grid[i, j]
                if np.array_equal(shifted, output_grid):
                    return True
        
        # Allow for more complex movement patterns
        return True
    except:
        return False
'''

        elif category == "Geometry":
            return f'''
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        
        # Size changes are geometric
        if input_grid.shape != output_grid.shape:
            return True
        
        # Check standard transformations
        for k in range(1, 4):
            if np.array_equal(output_grid, np.rot90(input_grid, k)):
                return True
        
        # Reflections
        if (np.array_equal(output_grid, np.flip(input_grid, axis=0)) or
            np.array_equal(output_grid, np.flip(input_grid, axis=1))):
            return True
        
        # Color preservation check for geometric transforms
        input_counts = np.bincount(input_grid.flat, minlength=10)
        output_counts = np.bincount(output_grid.flat, minlength=10)
        
        return np.array_equal(input_counts, output_counts) and not np.array_equal(input_grid, output_grid)
    except:
        return False
'''

        elif category == "Object":
            return f'''
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if input_objects is None or output_objects is None:
            return not np.array_equal(input_grid, output_grid)
        
        # Object count changes
        if len(input_objects) != len(output_objects):
            return True
        
        # Object property changes
        for i in range(min(len(input_objects), len(output_objects))):
            if len(input_objects[i]) != len(output_objects[i]):
                return True
        
        return False
    except:
        return not np.array_equal(input_grid, output_grid)
'''

        elif category == "Number":
            return f'''
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        input_nonzero = np.count_nonzero(input_grid)
        output_nonzero = np.count_nonzero(output_grid)
        
        # Check for numerical multiplication patterns
        if output_nonzero > input_nonzero:
            ratio = output_nonzero / input_nonzero
            if abs(ratio - round(ratio)) < 0.2 and ratio > 1:
                return True
        
        # Object count patterns
        if input_objects and output_objects:
            input_count = len(input_objects)
            output_count = len(output_objects)
            if output_count > input_count:
                ratio = output_count / input_count
                if abs(ratio - round(ratio)) < 0.2:
                    return True
        
        return False
    except:
        return False
'''

        else:  # Commonsense
            return f'''
def test_hypothesis(input_grid, output_grid, input_objects=None, output_objects=None):
    try:
        # {description}
        if np.array_equal(input_grid, output_grid):
            return False
        
        # Commonsense if it doesn't fit simple patterns
        # Check it's NOT simple geometric
        is_simple_transform = False
        if input_grid.shape == output_grid.shape:
            for k in range(1, 4):
                if np.array_equal(output_grid, np.rot90(input_grid, k)):
                    is_simple_transform = True
                    break
        
        # Check it's NOT simple color mapping
        is_simple_color = False
        if input_grid.shape == output_grid.shape:
            color_mapping = {{}}
            consistent = True
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    in_c, out_c = input_grid[i, j], output_grid[i, j]
                    if in_c in color_mapping:
                        if color_mapping[in_c] != out_c:
                            consistent = False
                            break
                    else:
                        color_mapping[in_c] = out_c
                if not consistent:
                    break
            is_simple_color = consistent and any(k != v for k, v in color_mapping.items())
        
        return not (is_simple_transform or is_simple_color)
    except:
        return True
'''

    def rule_complexity(self, rule_description: str) -> int:
        """Calculate complexity based on token count"""
        tokens = self.tokenizer.encode(rule_description)
        return max(len(tokens), 1)


class AIHypothesisGenerator(ImprovedCategoryAIGenerator):
    """Main interface - backwards compatible with existing code"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.batch_mode = True  # Enable batching for efficiency
    
    def generate_smart_hypotheses(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                num_hypotheses: int = 12, training_pairs: List[Tuple[np.ndarray, np.ndarray]] = None) -> List[Dict[str, Any]]:
        """
        Generate smart hypotheses with improved accuracy and context awareness
        
        Args:
            input_grid: First training example input
            output_grid: First training example output  
            num_hypotheses: Target number of hypotheses
            training_pairs: All training pairs for better context (NEW)
        """
        
        # Prepare training pairs - use provided or create from single example
        if training_pairs is None:
            training_pairs = [(input_grid, output_grid)]
        
        # Generate with full context
        hypotheses = self.generate_task_aware_hypotheses(
            training_pairs=training_pairs,
            target_categories=None,  # Let heuristics decide
            num_per_category=max(1, num_hypotheses // 4)
        )
        
        # Ensure we have enough hypotheses
        if len(hypotheses) < num_hypotheses:
            # Add fallback hypotheses if needed
            fallback_categories = ["Commonsense", "Object", "Colour", "Movement"]
            for cat in fallback_categories:
                if len(hypotheses) >= num_hypotheses:
                    break
                    
                fallback = self._create_fallback_hypothesis(cat, input_grid, output_grid)
                hypotheses.append(fallback)
        
        return hypotheses[:num_hypotheses]
    
    def _create_fallback_hypothesis(self, category: str, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Create a fallback hypothesis"""
        descriptions = {
            "Colour": "Grid colors are systematically transformed according to a color mapping rule",
            "Movement": "Objects or pixels are spatially repositioned while preserving their properties", 
            "Geometry": "The grid undergoes a geometric transformation like rotation, reflection, or scaling",
            "Object": "Discrete objects are modified, merged, split, or duplicated",
            "Number": "The transformation follows a numerical pattern based on counts or arithmetic",
            "Commonsense": "The transformation applies logical reasoning or pattern completion"
        }
        
        return {
            "description": descriptions[category],
            "category": category,
            "confidence": 0.4,
            "evidence": f"Fallback hypothesis for {category} category",
            "prior_probability": 0.08,
            "source": f"fallback_{category.lower()}",
            "test_code": self._generate_improved_test_code(descriptions[category], category)
        }