#!/usr/bin/env python3
"""
Path Simplify Extension for Inkscape
Simplifies paths by reducing control points while preserving shape
"""

import sys
import os
import math
import re
from copy import deepcopy

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Group, PathElement, Circle, Transform, TextElement

class PathSimplifyExtension(inkex.EffectExtension):
    """Extension to simplify paths by reducing control points"""
    
    def add_arguments(self, pars):
        """Parameter definitions"""
        # Simplification method
        pars.add_argument("--simplify_method", default="douglas_peucker")
        
        # Algorithm parameters
        pars.add_argument("--tolerance", type=float, default=2.0)
        pars.add_argument("--min_distance", type=float, default=1.0)
        pars.add_argument("--angle_threshold", type=float, default=5.0)
        
        # Advanced options
        pars.add_argument("--preserve_endpoints", type=bool, default=True)
        pars.add_argument("--preserve_corners", type=bool, default=True)
        pars.add_argument("--corner_threshold", type=float, default=30.0)
        
        # Reduction settings
        pars.add_argument("--max_reduction", type=float, default=0.8)
        pars.add_argument("--iterations", type=int, default=1)
        
        # Output options
        pars.add_argument("--create_copy", type=bool, default=True)
        pars.add_argument("--show_removed_points", type=bool, default=False)
        pars.add_argument("--show_statistics", type=bool, default=False)
    
    def effect(self):
        """Main processing method"""
        try:
            selected = self.svg.selected
            
            if not selected:
                self.msg("Please select one or more paths to simplify.")
                return
            
            current_layer = self.svg.get_current_layer()
            total_original = 0
            total_simplified = 0
            
            for element in selected.values():
                if isinstance(element, PathElement):
                    original_count, simplified_count = self.process_path(element, current_layer)
                    total_original += original_count
                    total_simplified += simplified_count
                else:
                    self.msg(f"Skipping non-path element: {element.tag}")
            
            if self.options.show_statistics and total_original > 0:
                reduction_percent = (1 - total_simplified / total_original) * 100
                stats_text = f"Points: {total_original} → {total_simplified} ({reduction_percent:.1f}% reduction)"
                self.add_statistics_text(stats_text, current_layer)
                    
        except Exception as e:
            self.msg(f"Error: {str(e)}")
    
    def process_path(self, path_element, layer):
        """Process a single path element"""
        try:
            # Parse path data
            path_data = path_element.get('d')
            if not path_data:
                self.msg("No path data found")
                return 0, 0
            
            # Debug: Show path data
            if len(path_data) > 100:
                self.msg(f"Path data: {path_data[:100]}...")
            else:
                self.msg(f"Path data: {path_data}")
            
            points = self.extract_points_from_path(path_data)
            
            self.msg(f"Extracted {len(points)} points from path")
            
            if len(points) < 3:
                self.msg(f"Path must have at least 3 points, but only found {len(points)} points")
                if len(points) > 0:
                    self.msg(f"Points found: {points}")
                return 0, 0
            
            original_count = len(points)
            
            simplified_points = points[:]
            for i in range(self.options.iterations):
                before_count = len(simplified_points)
                simplified_points = self.apply_simplification(simplified_points)
                after_count = len(simplified_points)
                
                self.msg(f"Iteration {i+1}: {before_count} → {after_count} points")
                
                reduction_ratio = 1 - len(simplified_points) / len(points)
                if reduction_ratio >= self.options.max_reduction:
                    break
            
            simplified_count = len(simplified_points)
            
            new_path_data = self.create_path_from_points(simplified_points)
            
            if self.options.show_removed_points:
                removed_points = self.find_removed_points(points, simplified_points)
                self.add_removed_points_markers(removed_points, layer)
            
            if self.options.create_copy:
                new_path = PathElement()
                new_path.set('d', new_path_data)
                
                # Copy style from original
                new_path.style = path_element.style.copy()
                
                transform = Transform(translate=(10, 10))
                new_path.transform = transform
                
                layer.add(new_path)
            else:
                path_element.set('d', new_path_data)
            
            return original_count, simplified_count
                
        except Exception as e:
            self.msg(f"Error processing path: {str(e)}")
            import traceback
            self.msg(f"Traceback: {traceback.format_exc()}")
            return 0, 0
    
    def extract_points_from_path(self, path_data):
        """Extract coordinate points from SVG path data using inkex parser"""
        try:
            from inkex.paths import Path
            
            path = Path(path_data)
            points = []
            
            for command in path:
                if hasattr(command, 'end_point'):
                    end_point = command.end_point(command.previous_end_point if hasattr(command, 'previous_end_point') else (0, 0))
                    if end_point is not None:
                        points.append((float(end_point[0]), float(end_point[1])))
                elif hasattr(command, 'args') and len(command.args) >= 2:
                    args = command.args
                    points.append((float(args[-2]), float(args[-1])))
            
            if len(points) < 2:
                points = self.manual_path_parse(path_data)
            
            if len(points) > 1:
                filtered_points = [points[0]]
                for i in range(1, len(points)):
                    if (abs(points[i][0] - filtered_points[-1][0]) > 0.01 or 
                        abs(points[i][1] - filtered_points[-1][1]) > 0.01):
                        filtered_points.append(points[i])
                points = filtered_points
            
            return points
            
        except Exception as e:
            # Fall back to manual parsing
            return self.manual_path_parse(path_data)
    
    def manual_path_parse(self, path_data):
        """Manual parsing of SVG path data as fallback"""
        points = []
        
        # Clean up the path data
        path_data = re.sub(r'([MLCZSQTAmlczsqta])', r' \1 ', path_data)
        path_data = re.sub(r',', ' ', path_data)
        path_data = re.sub(r'\s+', ' ', path_data)
        
        # Split into tokens
        tokens = path_data.strip().split()
        
        current_x, current_y = 0.0, 0.0
        start_x, start_y = 0.0, 0.0
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.upper() in 'MLCZSQTA':
                cmd = token
                i += 1
                
                coords = []
                while i < len(tokens) and not tokens[i].upper() in 'MLCZSQTA':
                    try:
                        coords.append(float(tokens[i]))
                    except ValueError:
                        break
                    i += 1
                i -= 1

                cmd_upper = cmd.upper()
                is_relative = cmd.islower()
                
                if cmd_upper == 'M':
                    for j in range(0, len(coords), 2):
                        if j + 1 < len(coords):
                            if is_relative and j > 0:
                                current_x += coords[j]
                                current_y += coords[j + 1]
                            else:
                                current_x = coords[j]
                                current_y = coords[j + 1]
                            
                            if j == 0:
                                start_x, start_y = current_x, current_y
                            
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'L':
                    for j in range(0, len(coords), 2):
                        if j + 1 < len(coords):
                            if is_relative:
                                current_x += coords[j]
                                current_y += coords[j + 1]
                            else:
                                current_x = coords[j]
                                current_y = coords[j + 1]
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'H':
                    for coord in coords:
                        if is_relative:
                            current_x += coord
                        else:
                            current_x = coord
                        points.append((current_x, current_y))
                        
                elif cmd_upper == 'V':
                    for coord in coords:
                        if is_relative:
                            current_y += coord
                        else:
                            current_y = coord
                        points.append((current_x, current_y))
                            
                elif cmd_upper == 'C':
                    for j in range(0, len(coords), 6):
                        if j + 5 < len(coords):
                            if is_relative:
                                current_x += coords[j + 4]
                                current_y += coords[j + 5]
                            else:
                                current_x = coords[j + 4]
                                current_y = coords[j + 5]
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'S':
                    for j in range(0, len(coords), 4):
                        if j + 3 < len(coords):
                            if is_relative:
                                current_x += coords[j + 2]
                                current_y += coords[j + 3]
                            else:
                                current_x = coords[j + 2]
                                current_y = coords[j + 3]
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'Q':
                    for j in range(0, len(coords), 4):
                        if j + 3 < len(coords):
                            if is_relative:
                                current_x += coords[j + 2]
                                current_y += coords[j + 3]
                            else:
                                current_x = coords[j + 2]
                                current_y = coords[j + 3]
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'T':
                    for j in range(0, len(coords), 2):
                        if j + 1 < len(coords):
                            if is_relative:
                                current_x += coords[j]
                                current_y += coords[j + 1]
                            else:
                                current_x = coords[j]
                                current_y = coords[j + 1]
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'A':
                    for j in range(0, len(coords), 7):
                        if j + 6 < len(coords):
                            if is_relative:
                                current_x += coords[j + 5]
                                current_y += coords[j + 6]
                            else:
                                current_x = coords[j + 5]
                                current_y = coords[j + 6]
                            points.append((current_x, current_y))
                            
                elif cmd_upper == 'Z':
                    if (current_x, current_y) != (start_x, start_y):
                        points.append((start_x, start_y))
                    current_x, current_y = start_x, start_y
            
            i += 1
        
        return points
    
    def apply_simplification(self, points):
        """Apply selected simplification method"""
        if self.options.simplify_method == "douglas_peucker":
            return self.douglas_peucker_simplify(points)
        elif self.options.simplify_method == "distance":
            return self.distance_simplify(points)
        elif self.options.simplify_method == "angle":
            return self.angle_simplify(points)
        elif self.options.simplify_method == "combined":
            return self.combined_simplify(points)
        elif self.options.simplify_method == "visvalingam":
            return self.visvalingam_simplify(points)
        else:
            return points
    
    def douglas_peucker_simplify(self, points):
        """Douglas-Peucker line simplification algorithm"""
        if len(points) < 3:
            return points
        
        return self.douglas_peucker_recursive(points, self.options.tolerance)
    
    def douglas_peucker_recursive(self, points, tolerance):
        """Recursive Douglas-Peucker implementation"""
        if len(points) < 3:
            return points
        
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(points) - 1):
            distance = self.point_to_line_distance(points[i], points[0], points[-1])
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        if max_distance > tolerance:
            if self.options.preserve_corners:
                angle = self.calculate_angle_at_point(points, max_index)
                if angle < math.radians(self.options.corner_threshold):
                    left_result = self.douglas_peucker_recursive(points[:max_index + 1], tolerance)
                    right_result = self.douglas_peucker_recursive(points[max_index:], tolerance)
                    return left_result[:-1] + right_result
            
            left_result = self.douglas_peucker_recursive(points[:max_index + 1], tolerance)
            right_result = self.douglas_peucker_recursive(points[max_index:], tolerance)
            
            return left_result[:-1] + right_result
        else:
            return [points[0], points[-1]]
    
    def distance_simplify(self, points):
        """Simplify based on minimum distance between points"""
        if len(points) < 3:
            return points
        
        simplified = [points[0]]
        
        for i in range(1, len(points) - 1):
            distance = self.calculate_distance(simplified[-1], points[i])
            
            if distance >= self.options.min_distance:
                simplified.append(points[i])
        
        if self.options.preserve_endpoints:
            simplified.append(points[-1])
        
        return simplified
    
    def angle_simplify(self, points):
        """Simplify based on angle changes between consecutive segments"""
        if len(points) < 3:
            return points
        
        simplified = [points[0]]  # Always keep first point
        
        for i in range(1, len(points) - 1):
            # Calculate angle change at this point
            angle = self.calculate_angle_at_point(points, i)
            angle_deg = math.degrees(angle)
            
            if angle_deg < (180 - self.options.angle_threshold):
                simplified.append(points[i])
        
        if self.options.preserve_endpoints:
            simplified.append(points[-1])
        
        return simplified
    
    def combined_simplify(self, points):
        """Combine distance and angle-based simplification"""
        distance_filtered = self.distance_simplify(points)
        
        angle_filtered = self.angle_simplify(distance_filtered)
        
        return angle_filtered
    
    def visvalingam_simplify(self, points):
        """Visvalingam-Whyatt algorithm (area-based simplification)"""
        if len(points) < 3:
            return points
        
        areas = []
        for i in range(1, len(points) - 1):
            area = self.triangle_area(points[i-1], points[i], points[i+1])
            areas.append((area, i))
        
        areas.sort()
        
        to_remove = set()
        for area, index in areas:
            if area < self.options.tolerance:
                to_remove.add(index)
            else:
                break
        
        simplified = []
        for i, point in enumerate(points):
            if i not in to_remove:
                simplified.append(point)
        
        return simplified
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        
        if line_length_sq == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)
        
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
        
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_angle_at_point(self, points, index):
        """Calculate angle at a specific point"""
        if index <= 0 or index >= len(points) - 1:
            return math.pi
        
        p1 = points[index - 1]
        p2 = points[index]
        p3 = points[index + 1]
        
        # Vector from p2 to p1
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Calculate magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return math.pi
        
        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        return math.acos(cos_angle)
    
    def triangle_area(self, p1, p2, p3):
        """Calculate area of triangle formed by three points"""
        return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)
    
    def find_removed_points(self, original, simplified):
        """Find points that were removed during simplification"""
        simplified_set = set(simplified)
        removed = [p for p in original if p not in simplified_set]
        return removed
    
    def create_path_from_points(self, points):
        """Create SVG path data from points"""
        if not points:
            return ""
        
        path_data = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
        
        for point in points[1:]:
            path_data.append(f"L {point[0]:.2f},{point[1]:.2f}")
        
        return " ".join(path_data)
    
    def add_removed_points_markers(self, points, layer):
        """Add visual markers for removed points"""
        if not points:
            return
            
        group = Group()
        group.label = "Removed Points"
        
        for x, y in points:
            circle = Circle()
            circle.set('cx', str(x))
            circle.set('cy', str(y))
            circle.set('r', '1.5')
            circle.style = {
                'fill': 'red',
                'stroke': 'none',
                'opacity': '0.6'
            }
            group.add(circle)
        
        layer.add(group)
    
    def add_statistics_text(self, stats_text, layer):
        """Add statistics text to the layer"""
        text = TextElement()
        text.text = stats_text
        text.set('x', '10')
        text.set('y', '30')
        text.style = {
            'font-family': 'Arial',
            'font-size': '12px',
            'fill': 'black'
        }
        
        layer.add(text)

if __name__ == '__main__':
    try:
        extension = PathSimplifyExtension()
        extension.run()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)