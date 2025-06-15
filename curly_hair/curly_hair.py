#!/usr/bin/env python3
"""
Curly Hair Generator Extension for Inkscape - New Enhanced Version
"""

import sys
import os
import math
import random

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Group, PathElement

class CurlyHairExtension(inkex.EffectExtension):
    """Extension to generate curly hair, whiskers, and body hair"""
    
    def add_arguments(self, pars):
        """Parameter definitions"""
        # Hair/strand count
        pars.add_argument("--strand_count", type=int, default=10)
        
        # Length parameters
        pars.add_argument("--length_min", type=float, default=50.0)
        pars.add_argument("--length_max", type=float, default=150.0)
        
        # Thickness parameters
        pars.add_argument("--thickness_start", type=float, default=3.0)
        pars.add_argument("--thickness_end", type=float, default=0.2)
        pars.add_argument("--taper_position", type=float, default=0.3)
        pars.add_argument("--taper_curve", default="ease_out")
        
        # Curl parameters
        pars.add_argument("--curl_strength", type=float, default=1.0)
        pars.add_argument("--curl_frequency", type=float, default=2.0)
        pars.add_argument("--angle_variation", type=float, default=45.0)
        pars.add_argument("--curl_randomness", type=float, default=0.5)
        
        # Direction parameters
        pars.add_argument("--base_direction", type=float, default=270.0)
        pars.add_argument("--direction_spread", type=float, default=30.0)
        
        # Positioning
        pars.add_argument("--start_area_width", type=float, default=100.0)
        pars.add_argument("--start_area_height", type=float, default=20.0)
        pars.add_argument("--center_x", type=float, default=200.0)
        pars.add_argument("--center_y", type=float, default=200.0)
        
        # Advanced curl options (with improved defaults)
        pars.add_argument("--segments_per_curl", type=int, default=8) 
        pars.add_argument("--curl_decay", type=float, default=0.2)
        pars.add_argument("--spiral_tendency", type=float, default=0.5)
        
        # Smoothing options
        pars.add_argument("--smooth_curves", type=bool, default=True)
        pars.add_argument("--smoothing_factor", type=float, default=0.5)
        
        # Hair type presets
        pars.add_argument("--hair_type", default="custom")
        
        # Style
        pars.add_argument("--render_mode", default="stroke")
    
    def effect(self):
        """Main processing method"""
        try:
            current_layer = self.svg.get_current_layer()
            
            self.apply_hair_preset()
            
            main_group = Group()
            main_group.label = "Curly Hair"
            
            for i in range(self.options.strand_count):
                strand = self.create_hair_strand(i)
                if strand is not None:
                    main_group.add(strand)
            
            current_layer.add(main_group)
            
        except Exception as e:
            self.msg(f"Error: {str(e)}")
    
    def apply_hair_preset(self):
        """Apply preset parameters for different hair types"""
        presets = {
            "loose_curls": {
                "curl_strength": 0.8,
                "curl_frequency": 1.5,
                "angle_variation": 30.0,
                "segments_per_curl": 10,
                "curl_decay": 0.1
            },
            "tight_curls": {
                "curl_strength": 1.5,
                "curl_frequency": 4.0,
                "angle_variation": 60.0,
                "segments_per_curl": 8,
                "curl_decay": 0.3
            },
            "wavy": {
                "curl_strength": 0.5,
                "curl_frequency": 1.0,
                "angle_variation": 20.0,
                "segments_per_curl": 12,
                "curl_decay": 0.05
            },
            "kinky": {
                "curl_strength": 1.8,
                "curl_frequency": 5.0,
                "angle_variation": 70.0,
                "segments_per_curl": 6,
                "curl_decay": 0.3,
                "thickness_end": 0.05
            },
            "mustache": {
                "curl_strength": 1.2,
                "curl_frequency": 3.0,
                "angle_variation": 40.0,
                "thickness_start": 2.0,
                "thickness_end": 0.1,
                "length_min": 20.0,
                "length_max": 60.0,
                "segments_per_curl": 8
            },
            "eyelashes": {
                "curl_strength": 0.3,
                "curl_frequency": 0.5,
                "angle_variation": 15.0,
                "thickness_start": 1.5,
                "thickness_end": 0.02,
                "length_min": 15.0,
                "length_max": 25.0,
                "segments_per_curl": 6
            },
            "fur": {
                "curl_strength": 0.7,
                "curl_frequency": 2.5,
                "angle_variation": 35.0,
                "thickness_start": 1.0,
                "thickness_end": 0.05,
                "curl_randomness": 0.8,
                "segments_per_curl": 8
            }
        }
        
        if self.options.hair_type in presets:
            preset = presets[self.options.hair_type]
            for key, value in preset.items():
                if hasattr(self.options, key):
                    setattr(self.options, key, value)
    
    def create_hair_strand(self, strand_index):
        """Create a single hair strand"""
        start_x = (self.options.center_x - self.options.start_area_width/2 + 
                  random.uniform(0, self.options.start_area_width))
        start_y = (self.options.center_y - self.options.start_area_height/2 + 
                  random.uniform(0, self.options.start_area_height))
        
        strand_length = random.uniform(self.options.length_min, self.options.length_max)
        
        base_angle = math.radians(self.options.base_direction + 
                                 random.uniform(-self.options.direction_spread/2, 
                                              self.options.direction_spread/2))
        
        points, thicknesses = self.generate_strand_path(start_x, start_y, base_angle, strand_length)
        
        if self.options.render_mode == "filled":
            path_data = self.create_tapered_path(points, thicknesses)
            fill_color = '#000000'
            stroke_color = 'none'
            stroke_width = '0'
        elif self.options.render_mode == "stroke":
            return self.create_multi_segment_stroke(points, thicknesses)
        else:
            path_data = self.create_center_line_path(points)
            fill_color = 'none'
            stroke_color = '#000000'
            avg_thickness = (self.options.thickness_start + self.options.thickness_end) / 2
            stroke_width = str(avg_thickness)
        
        path_element = PathElement()
        path_element.set('d', path_data)
        
        path_element.style = {
            'fill': fill_color,
            'stroke': stroke_color,
            'stroke-width': stroke_width,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
            'opacity': '1.0'
        }
        
        return path_element
    
    def create_multi_segment_stroke(self, points, thicknesses):
        """Create multiple stroke segments with variable width"""
        if len(points) < 2:
            return None
        
        group = Group()
        group.label = "Variable Stroke Hair"
        
        num_segments = min(len(points) - 1, 8)
        segment_length = len(points) // num_segments
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length + 1, len(points))
            
            if end_idx <= start_idx + 1:
                continue
            
            segment_points = points[start_idx:end_idx]
            
            thickness_start_idx = start_idx
            thickness_end_idx = min(end_idx - 1, len(thicknesses) - 1)
            
            if thickness_start_idx < len(thicknesses) and thickness_end_idx < len(thicknesses):
                avg_thickness = (thicknesses[thickness_start_idx] + thicknesses[thickness_end_idx]) / 2
            else:
                avg_thickness = self.options.thickness_start
            
            path_data = self.create_center_line_path(segment_points)
            
            path_element = PathElement()
            path_element.set('d', path_data)
            path_element.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': str(max(0.01, avg_thickness)),
                'stroke-linecap': 'round',
                'stroke-linejoin': 'round',
                'opacity': '1.0'
            }
            
            group.add(path_element)
        
        return group
    
    def generate_strand_path(self, start_x, start_y, base_angle, total_length):
        """Generate the path points and thicknesses for a hair strand"""
        points = [(start_x, start_y)]
        thicknesses = [self.options.thickness_start]
        
        segments_per_curl = max(6, self.options.segments_per_curl)
        segment_length = total_length / (segments_per_curl * self.options.curl_frequency)
        num_segments = int(total_length / segment_length)
        
        current_x, current_y = start_x, start_y
        current_angle = base_angle
        
        for i in range(1, num_segments + 1):
            progress = i / num_segments
            
            curl_phase = progress * self.options.curl_frequency * 2 * math.pi
            curl_amplitude = self.options.curl_strength * (1 - self.options.curl_decay * progress)
            
            spiral_offset = self.options.spiral_tendency * progress * math.pi
            
            curl_deviation = (curl_amplitude * math.sin(curl_phase + spiral_offset) * 
                            math.radians(self.options.angle_variation))
            
            random_deviation = (random.uniform(-1, 1) * self.options.curl_randomness * 
                              math.radians(self.options.angle_variation * 0.5))
            
            current_angle += curl_deviation + random_deviation
            
            current_x += segment_length * math.cos(current_angle)
            current_y += segment_length * math.sin(current_angle)
            
            points.append((current_x, current_y))
            
            thickness = self.calculate_thickness(progress)
            thicknesses.append(thickness)
        
        return points, thicknesses
    
    def calculate_thickness(self, progress):
        """Calculate thickness at given progress along the strand"""
        taper_start = self.options.taper_position
        
        if progress <= taper_start:
            return self.options.thickness_start
        
        # Normalize progress in taper region
        taper_progress = (progress - taper_start) / (1.0 - taper_start)
        
        # Apply taper curve
        if self.options.taper_curve == "linear":
            factor = 1.0 - taper_progress
        elif self.options.taper_curve == "ease_out":
            factor = 1.0 - taper_progress**2
        elif self.options.taper_curve == "ease_in":
            factor = 1.0 - math.sqrt(taper_progress)
        elif self.options.taper_curve == "sigmoid":
            x = (taper_progress - 0.5) * 6
            sigmoid = 1 / (1 + math.exp(-x))
            factor = 1.0 - sigmoid
        else:
            factor = 1.0 - taper_progress
        
        return (self.options.thickness_start * factor + 
                self.options.thickness_end * (1.0 - factor))
    
    def create_tapered_path(self, points, thicknesses):
        """Create SVG path with variable thickness and improved tip"""
        if len(points) < 2:
            return ""
        
        left_points = []
        right_points = []
        
        for i, (x, y) in enumerate(points):
            thickness = thicknesses[i] / 2
            
            if i == 0:
                # Use direction to next point
                if len(points) > 1:
                    dx = points[1][0] - points[0][0]
                    dy = points[1][1] - points[0][1]
                else:
                    dx, dy = 1, 0
            elif i == len(points) - 1:
                # Use direction from previous point
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
            else:
                # Use average direction
                dx1 = points[i][0] - points[i-1][0]
                dy1 = points[i][1] - points[i-1][1]
                dx2 = points[i+1][0] - points[i][0]
                dy2 = points[i+1][1] - points[i][1]
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2
            
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                perp_x = -dy / length * thickness
                perp_y = dx / length * thickness
            else:
                perp_x = perp_y = 0
            
            left_points.append((x + perp_x, y + perp_y))
            right_points.append((x - perp_x, y - perp_y))
        
        return self.create_improved_tapered_path(left_points, right_points, points)
    
    def create_improved_tapered_path(self, left_points, right_points, center_points):
        """Create tapered path with very sharp tip to prevent closed shapes"""
        path_parts = []
        
        if left_points:
            path_parts.append(f"M {left_points[0][0]:.2f},{left_points[0][1]:.2f}")
            for point in left_points[1:]:
                path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
        
        if center_points and len(center_points) > 0:
            tip_x = center_points[-1][0]
            tip_y = center_points[-1][1]
            path_parts.append(f"L {tip_x:.2f},{tip_y:.2f}")
            
            if len(right_points) > 2:
                skip_points = min(2, len(right_points) // 4)
                for point in reversed(right_points[:-skip_points]):
                    path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
            elif right_points:
                for point in reversed(right_points[:-1]):
                    path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
        
        # Close path
        path_parts.append("Z")
        
        return " ".join(path_parts)
    
    def create_center_line_path(self, points):
        """Create simple center line path"""
        if len(points) < 2:
            return ""
        
        path_parts = []
        path_parts.append(f"M {points[0][0]:.2f},{points[0][1]:.2f}")
        
        if self.options.smooth_curves and len(points) > 3:
            # Create smooth curve
            smoothing = self.options.smoothing_factor
            
            for i in range(len(points) - 1):
                current = points[i]
                next_point = points[i + 1]
                
                if i == 0:
                    prev_point = current
                else:
                    prev_point = points[i - 1]
                
                if i + 2 < len(points):
                    after_next = points[i + 2]
                else:
                    after_next = next_point
                
                # Calculate control points
                cp1_x = current[0] + (next_point[0] - prev_point[0]) * smoothing * 0.5
                cp1_y = current[1] + (next_point[1] - prev_point[1]) * smoothing * 0.5
                
                cp2_x = next_point[0] - (after_next[0] - current[0]) * smoothing * 0.5
                cp2_y = next_point[1] - (after_next[1] - current[1]) * smoothing * 0.5
                
                path_parts.append(f"C {cp1_x:.2f},{cp1_y:.2f} {cp2_x:.2f},{cp2_y:.2f} {next_point[0]:.2f},{next_point[1]:.2f}")
        else:
            for point in points[1:]:
                path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
        
        return " ".join(path_parts)

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
    <g id="layer1"></g>
</svg>'''
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(test_svg)
                sys.argv.append(f.name)
        
        extension = CurlyHairExtension()
        extension.run()
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)