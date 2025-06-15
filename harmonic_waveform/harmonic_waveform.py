#!/usr/bin/env python3
"""
Harmonic Waveform Generator Extension for Inkscape
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

class HarmonicWaveformExtension(inkex.EffectExtension):
    """Extension to generate harmonic additive synthesis waveforms"""
    
    def add_arguments(self, pars):
        """Parameter definitions"""
        # Basic wave settings
        pars.add_argument("--fundamental_freq", type=float, default=440.0)
        pars.add_argument("--wave_cycles", type=float, default=2.0)
        pars.add_argument("--resolution", type=int, default=500)
        
        # Harmonic control
        pars.add_argument("--harmonic_count", type=int, default=8)
        pars.add_argument("--decay_pattern", default="harmonic")
        pars.add_argument("--decay_strength", type=float, default=1.0)
        
        # Display settings
        pars.add_argument("--wave_display", default="full")
        
        # Smoothing settings
        pars.add_argument("--enable_smoothing", type=inkex.Boolean, default=True)
        pars.add_argument("--smoothing_method", default="spline")
        pars.add_argument("--smoothing_strength", type=float, default=1.0)
        
        # Path optimization settings
        pars.add_argument("--enable_optimization", type=inkex.Boolean, default=False)
        pars.add_argument("--optimization_method", default="douglas_peucker")
        pars.add_argument("--optimization_tolerance", type=float, default=1.0)
        
        # Style settings
        pars.add_argument("--line_thickness", type=float, default=2.0)
        pars.add_argument("--line_color", default="#000000ff")
        
        # Canvas settings
        pars.add_argument("--canvas_width", type=float, default=800.0)
        pars.add_argument("--canvas_height", type=float, default=200.0)
        
        # Amplitude settings
        pars.add_argument("--amplitude_scale", type=float, default=0.8)
    
    def effect(self):
        """Main processing method"""
        try:
            current_layer = self.svg.get_current_layer()
            
            main_group = Group()
            main_group.label = "Harmonic Waveform"
            
            # Generate waveform data
            waveform_data = self.generate_waveform()
            original_point_count = len(waveform_data)
            
            # Apply smoothing if enabled
            if self.options.enable_smoothing:
                waveform_data = self.apply_smoothing(waveform_data)
            
            # Apply path optimization if enabled
            if self.options.enable_optimization:
                waveform_data = self.optimize_path(waveform_data)
                optimized_point_count = len(waveform_data)
                
                # Add comment to SVG showing optimization results
                reduction_percent = ((original_point_count - optimized_point_count) / original_point_count) * 100
                main_group.set('data-original-points', str(original_point_count))
                main_group.set('data-optimized-points', str(optimized_point_count))
                main_group.set('data-reduction-percent', f'{reduction_percent:.1f}%')
            
            # Create SVG path
            waveform_path = self.create_waveform_path(waveform_data)
            if waveform_path is not None:
                main_group.add(waveform_path)
            
            current_layer.add(main_group)
            
        except Exception as e:
            pass
    
    def generate_waveform(self):
        """Generate waveform data using harmonic additive synthesis"""
        points = []
        
        # Calculate resolution - use higher resolution for smoothing
        base_resolution = self.options.resolution
        if self.options.enable_smoothing and self.options.smoothing_method == "high_res":
            actual_resolution = int(base_resolution * self.options.smoothing_strength)
        else:
            actual_resolution = base_resolution
        
        # Calculate time parameters
        total_time = self.options.wave_cycles / self.options.fundamental_freq
        time_step = total_time / actual_resolution
        
        # Generate harmonic amplitudes based on decay pattern
        harmonic_amplitudes = self.calculate_harmonic_amplitudes()
        
        # Generate waveform points
        for i in range(actual_resolution + 1):
            t = i * time_step
            
            # Calculate composite waveform by adding harmonics
            amplitude = 0.0
            
            for harmonic_num in range(1, self.options.harmonic_count + 1):
                harmonic_freq = self.options.fundamental_freq * harmonic_num
                harmonic_amplitude = harmonic_amplitudes[harmonic_num - 1]
                
                # Add this harmonic to the composite waveform
                amplitude += harmonic_amplitude * math.sin(2 * math.pi * harmonic_freq * t)
            
            # Apply amplitude scaling
            amplitude *= self.options.amplitude_scale
            
            # Apply wave display filter
            if self.options.wave_display == "positive" and amplitude < 0:
                amplitude = 0
            elif self.options.wave_display == "negative" and amplitude > 0:
                amplitude = 0
            
            # Convert to canvas coordinates
            x = (i / actual_resolution) * self.options.canvas_width
            y = (self.options.canvas_height / 2) - (amplitude * self.options.canvas_height / 4)
            
            points.append((x, y))
        
        return points
    
    def apply_smoothing(self, points):
        """Apply smoothing to waveform points"""
        if not points or len(points) < 3:
            return points
        
        if self.options.smoothing_method == "spline":
            return self.smooth_with_spline(points)
        elif self.options.smoothing_method == "bezier":
            return self.smooth_with_bezier(points)
        elif self.options.smoothing_method == "high_res":
            # High resolution smoothing is handled in generation
            return self.downsample_points(points)
        elif self.options.smoothing_method == "linear_average":
            return self.smooth_with_linear_average(points)
        else:
            return points
    
    def smooth_with_spline(self, points):
        """Apply spline-based smoothing"""
        # Simple spline-like smoothing using weighted averaging
        smoothed_points = []
        
        for i in range(len(points)):
            if i == 0 or i == len(points) - 1:
                # Keep first and last points unchanged
                smoothed_points.append(points[i])
            else:
                # Apply weighted average
                weight = 1.0 / self.options.smoothing_strength
                
                prev_point = points[i - 1]
                curr_point = points[i]
                next_point = points[i + 1]
                
                # Smooth Y coordinate (keep X unchanged for proper timing)
                smooth_y = (prev_point[1] * weight + curr_point[1] * (1 - 2 * weight) + next_point[1] * weight)
                smoothed_points.append((curr_point[0], smooth_y))
        
        return smoothed_points
    
    def optimize_path(self, points):
        """Apply path optimization to reduce point count while preserving shape"""
        if not points or len(points) < 3:
            return points
        
        if self.options.optimization_method == "douglas_peucker":
            return self.douglas_peucker_optimize(points)
        elif self.options.optimization_method == "adaptive":
            return self.adaptive_sampling_optimize(points)
        elif self.options.optimization_method == "curvature":
            return self.curvature_based_optimize(points)
        elif self.options.optimization_method == "threshold":
            return self.threshold_optimize(points)
        else:
            return points
    
    def douglas_peucker_optimize(self, points):
        """Douglas-Peucker line simplification algorithm"""
        if len(points) <= 2:
            return points
        
        tolerance = self.options.optimization_tolerance
        
        def perpendicular_distance(point, line_start, line_end):
            """Calculate perpendicular distance from point to line"""
            if line_start == line_end:
                return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
            
            # Line equation: ax + by + c = 0
            a = line_end[1] - line_start[1]
            b = line_start[0] - line_end[0]
            c = line_end[0] * line_start[1] - line_start[0] * line_end[1]
            
            # Distance formula
            distance = abs(a * point[0] + b * point[1] + c) / math.sqrt(a*a + b*b)
            return distance
        
        def douglas_peucker_recursive(points_subset, start_idx, end_idx):
            """Recursive Douglas-Peucker implementation"""
            if end_idx <= start_idx + 1:
                return [start_idx, end_idx]
            
            # Find the point with maximum distance from line
            max_distance = 0
            max_index = start_idx
            
            for i in range(start_idx + 1, end_idx):
                distance = perpendicular_distance(points_subset[i], points_subset[start_idx], points_subset[end_idx])
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            
            # If max distance is greater than tolerance, recursively simplify
            if max_distance > tolerance:
                # Recursive call for first half
                left_indices = douglas_peucker_recursive(points_subset, start_idx, max_index)
                # Recursive call for second half
                right_indices = douglas_peucker_recursive(points_subset, max_index, end_idx)
                
                # Combine results (remove duplicate middle point)
                return left_indices[:-1] + right_indices
            else:
                # All points between start and end can be removed
                return [start_idx, end_idx]
        
        # Apply Douglas-Peucker algorithm
        keep_indices = douglas_peucker_recursive(points, 0, len(points) - 1)
        simplified_points = [points[i] for i in sorted(set(keep_indices))]
        
        return simplified_points
    
    def adaptive_sampling_optimize(self, points):
        """Adaptive sampling based on local curvature"""
        if len(points) < 3:
            return points
        
        optimized_points = [points[0]]  # Always keep first point
        
        i = 1
        tolerance = self.options.optimization_tolerance
        
        while i < len(points) - 1:
            current_point = points[i]
            prev_point = optimized_points[-1]
            
            # Look ahead to find a good sampling point
            look_ahead = min(int(tolerance * 5), len(points) - i - 1)
            
            # Calculate curvature in the look-ahead window
            max_curvature = 0
            best_next_index = i + 1
            
            for j in range(1, look_ahead + 1):
                if i + j >= len(points):
                    break
                
                next_point = points[i + j]
                curvature = self.calculate_curvature(prev_point, current_point, next_point)
                
                if curvature > max_curvature:
                    max_curvature = curvature
                    best_next_index = i + j
            
            # Decide whether to keep the current point
            if max_curvature > tolerance * 0.1:  # Adjust threshold based on tolerance
                optimized_points.append(current_point)
                i += 1
            else:
                # Skip to the best next point
                i = best_next_index
        
        optimized_points.append(points[-1])  # Always keep last point
        return optimized_points
    
    def curvature_based_optimize(self, points):
        """Remove points with low curvature"""
        if len(points) < 3:
            return points
        
        optimized_points = [points[0]]  # Always keep first point
        tolerance = self.options.optimization_tolerance * 0.01
        
        for i in range(1, len(points) - 1):
            prev_point = points[i - 1]
            current_point = points[i]
            next_point = points[i + 1]
            
            curvature = self.calculate_curvature(prev_point, current_point, next_point)
            
            # Keep points with high curvature
            if curvature > tolerance:
                optimized_points.append(current_point)
        
        optimized_points.append(points[-1])  # Always keep last point
        return optimized_points
    
    def threshold_optimize(self, points):
        """Remove points that are too close to each other"""
        if len(points) < 2:
            return points
        
        optimized_points = [points[0]]  # Always keep first point
        threshold = self.options.optimization_tolerance
        
        for point in points[1:]:
            last_kept_point = optimized_points[-1]
            distance = math.sqrt((point[0] - last_kept_point[0])**2 + (point[1] - last_kept_point[1])**2)
            
            if distance >= threshold:
                optimized_points.append(point)
        
        # Ensure we keep the last point if it's not already kept
        if optimized_points[-1] != points[-1]:
            optimized_points.append(points[-1])
        
        return optimized_points
    
    def calculate_curvature(self, p1, p2, p3):
        """Calculate curvature at point p2 given three consecutive points"""
        # Vector from p1 to p2
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        # Vector from p2 to p3  
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Lengths of vectors
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 == 0 or len2 == 0:
            return 0
        
        # Normalize vectors
        v1_norm = (v1[0] / len1, v1[1] / len1)
        v2_norm = (v2[0] / len2, v2[1] / len2)
        
        # Calculate angle between vectors using dot product
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        dot_product = max(-1, min(1, dot_product))  # Clamp to [-1, 1]
        
        angle = math.acos(dot_product)
        return angle  # Curvature is proportional to angle change
    
    def smooth_with_bezier(self, points):
        """Apply Bezier curve smoothing"""
        # For Bezier smoothing, we'll reduce the number of points and create smoother curves
        if len(points) < 4:
            return points
        
        # Sample every nth point based on smoothing strength
        step = max(1, int(self.options.smoothing_strength))
        sampled_points = points[::step]
        
        # Ensure we include the last point
        if points[-1] not in sampled_points:
            sampled_points.append(points[-1])
        
        return sampled_points
    
    def downsample_points(self, points):
        """Downsample high-resolution points to target resolution"""
        target_resolution = self.options.resolution
        if len(points) <= target_resolution:
            return points
        
        # Sample points evenly
        downsampled = []
        step = len(points) / target_resolution
        
        for i in range(target_resolution + 1):
            index = min(int(i * step), len(points) - 1)
            downsampled.append(points[index])
        
        return downsampled
    
    def smooth_with_linear_average(self, points):
        """Apply linear average smoothing - flattens waveform towards center"""
        if not points or len(points) < 3:
            return points
            
        smoothed_points = []
        
        # Calculate the average Y position (center line)
        total_y = sum(point[1] for point in points)
        average_y = total_y / len(points)
        
        # Calculate center Y of canvas for reference
        center_y = self.options.canvas_height / 2
        
        for i, (x, y) in enumerate(points):
            if i == 0 or i == len(points) - 1:
                # Keep first and last points unchanged
                smoothed_points.append((x, y))
            else:
                # Blend towards the center based on smoothing strength
                blend_factor = 1.0 / self.options.smoothing_strength
                
                # Choose target: average Y or center Y based on preference
                target_y = average_y  # Use average height of waveform
                
                # Linear interpolation towards target
                smooth_y = y * (1 - blend_factor) + target_y * blend_factor
                smoothed_points.append((x, smooth_y))
        
        return smoothed_points
    
    def calculate_harmonic_amplitudes(self):
        """Calculate amplitude for each harmonic based on decay pattern"""
        amplitudes = []
        
        for n in range(1, self.options.harmonic_count + 1):
            if self.options.decay_pattern == "none":
                # No harmonics, fundamental only
                amplitude = 1.0 if n == 1 else 0.0
                
            elif self.options.decay_pattern == "linear":
                # Linear decay: 1, 0.8, 0.6, 0.4...
                amplitude = max(0.0, 1.0 - (n - 1) * self.options.decay_strength / self.options.harmonic_count)
                
            elif self.options.decay_pattern == "exponential":
                # Exponential decay: 1, 0.5, 0.25, 0.125...
                amplitude = (0.5) ** ((n - 1) * self.options.decay_strength)
                
            elif self.options.decay_pattern == "harmonic":
                # Harmonic series: 1, 1/2, 1/3, 1/4...
                amplitude = (1.0 / n) ** self.options.decay_strength
                
            elif self.options.decay_pattern == "sawtooth":
                # Sawtooth approximation: 1, -1/2, 1/3, -1/4...
                amplitude = ((-1) ** (n - 1)) / (n ** self.options.decay_strength)
                
            elif self.options.decay_pattern == "square":
                # Square wave approximation: 1, 0, 1/3, 0, 1/5...
                if n % 2 == 1:  # Odd harmonics only
                    amplitude = (1.0 / n) ** self.options.decay_strength
                else:
                    amplitude = 0.0
                    
            elif self.options.decay_pattern == "organ":
                # Organ-like: Strong odd harmonics, weak even
                if n % 2 == 1:  # Odd harmonics
                    amplitude = (1.0 / n) ** (self.options.decay_strength * 0.5)
                else:  # Even harmonics
                    amplitude = (1.0 / n) ** (self.options.decay_strength * 2.0)
                    
            elif self.options.decay_pattern == "brass":
                # Brass instruments: Strong mid-range harmonics
                peak_harmonic = 5  # Peak around 5th harmonic
                distance = abs(n - peak_harmonic)
                amplitude = math.exp(-distance * self.options.decay_strength * 0.3) / n ** 0.3
                
            elif self.options.decay_pattern == "string":
                # String instruments: Natural decay with some peaks
                base_amplitude = (1.0 / n) ** self.options.decay_strength
                # Add peaks at specific harmonics (3rd, 5th, 7th)
                if n in [3, 5, 7, 9]:
                    base_amplitude *= 1.5
                amplitude = base_amplitude
                
            elif self.options.decay_pattern == "formant":
                # Formant-like: Peaks at vocal formant frequencies
                # Simulate formants at harmonics 2-4, 8-12, 16-20
                formant_regions = [(2, 4), (8, 12), (16, 20)]
                amplitude = (1.0 / n) ** (self.options.decay_strength * 2.0)
                
                for start, end in formant_regions:
                    if start <= n <= end:
                        amplitude *= 3.0  # Boost formant regions
                        
            elif self.options.decay_pattern == "random":
                # Random distribution
                random.seed(42 + n)  # Reproducible randomness
                amplitude = random.uniform(0.0, 1.0) / (n ** (self.options.decay_strength * 0.5))
                
            elif self.options.decay_pattern == "fibonacci":
                # Fibonacci sequence influence
                fib_nums = self.fibonacci_sequence(min(n, 20))  # Limit for performance
                if n <= len(fib_nums):
                    fib_factor = fib_nums[n - 1] / fib_nums[-1]  # Normalize
                else:
                    fib_factor = 1.0 / n
                amplitude = fib_factor / (n ** (self.options.decay_strength * 0.5))
                
            elif self.options.decay_pattern == "prime":
                # Prime number peaks
                if self.is_prime(n):
                    amplitude = (1.0 / n) ** (self.options.decay_strength * 0.5)  # Strong primes
                else:
                    amplitude = (1.0 / n) ** (self.options.decay_strength * 1.5)  # Weak non-primes
                    
            elif self.options.decay_pattern == "gaussian":
                # Gaussian (bell curve) distribution
                center = min(8, self.options.harmonic_count // 4)  # Peak around 8th harmonic
                sigma = self.options.harmonic_count / (6 * self.options.decay_strength)
                amplitude = math.exp(-((n - center) ** 2) / (2 * sigma ** 2))
            
            else:
                # Default to harmonic
                amplitude = 1.0 / n
            
            amplitudes.append(amplitude)
        
        return amplitudes
    
    def fibonacci_sequence(self, n):
        """Generate Fibonacci sequence up to n terms"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def is_prime(self, n):
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def create_waveform_path(self, points):
        """Create SVG path from waveform points"""
        if not points:
            return None
        
        # Build path data based on smoothing method
        if (self.options.enable_smoothing and 
            self.options.smoothing_method == "bezier" and 
            len(points) >= 4):
            path_data = self.create_bezier_path(points)
        else:
            path_data = self.create_linear_path(points)
        
        # Create path element
        path_element = PathElement()
        path_element.set('d', " ".join(path_data))
        
        # Convert color format - fix for transparency issue
        color = self.options.line_color
        
        # Handle different color formats
        if isinstance(color, str):
            if len(color) == 9 and color.startswith('#'):  # #RRGGBBaa format
                color = color[:7]  # Remove alpha component
            elif len(color) == 7 and color.startswith('#'):  # #RRGGBB format
                pass  # Already correct
            elif not color.startswith('#'):
                color = '#000000'  # Default to black if malformed
        else:
            color = '#000000'  # Default to black
        
        # Ensure color is valid
        if not color or color == '#':
            color = '#000000'
        
        # Set style with explicit opacity
        path_element.style = {
            'fill': 'none',
            'stroke': color,
            'stroke-width': f'{self.options.line_thickness}px',
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
            'stroke-opacity': '1.0',
            'opacity': '1.0'
        }
        
        return path_element
    
    def create_linear_path(self, points):
        """Create linear path data"""
        path_data = []
        
        # Move to first point
        path_data.append(f"M {points[0][0]:.2f},{points[0][1]:.2f}")
        
        # Line to subsequent points
        for x, y in points[1:]:
            path_data.append(f"L {x:.2f},{y:.2f}")
        
        return path_data
    
    def create_bezier_path(self, points):
        """Create smooth Bezier curve path data"""
        path_data = []
        
        # Move to first point
        path_data.append(f"M {points[0][0]:.2f},{points[0][1]:.2f}")
        
        # Create smooth curves between points
        for i in range(1, len(points)):
            if i == 1:
                # First curve - use quadratic
                cp_x = (points[0][0] + points[1][0]) / 2
                cp_y = (points[0][1] + points[1][1]) / 2
                path_data.append(f"Q {cp_x:.2f},{cp_y:.2f} {points[1][0]:.2f},{points[1][1]:.2f}")
            else:
                # Subsequent curves - use smooth cubic curves
                prev_point = points[i - 1]
                curr_point = points[i]
                
                # Calculate control points for smooth curve
                control_distance = 0.3  # Adjust for curve smoothness
                
                if i < len(points) - 1:
                    next_point = points[i + 1]
                    # Control point based on direction to next point
                    dx = next_point[0] - prev_point[0]
                    dy = next_point[1] - prev_point[1]
                else:
                    # Last point
                    dx = curr_point[0] - prev_point[0]
                    dy = curr_point[1] - prev_point[1]
                
                cp1_x = prev_point[0] + dx * control_distance
                cp1_y = prev_point[1] + dy * control_distance
                cp2_x = curr_point[0] - dx * control_distance
                cp2_y = curr_point[1] - dy * control_distance
                
                path_data.append(f"C {cp1_x:.2f},{cp1_y:.2f} {cp2_x:.2f},{cp2_y:.2f} {curr_point[0]:.2f},{curr_point[1]:.2f}")
        
        return path_data

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="200" viewBox="0 0 800 200">
    <g id="layer1"></g>
</svg>'''
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(test_svg)
                sys.argv.append(f.name)
        
        extension = HarmonicWaveformExtension()
        extension.run()
        
    except Exception as e:
        pass