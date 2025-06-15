#!/usr/bin/env python3
"""
Lissajous Curves Generator Extension for Inkscape
"""

import sys
import os
import math

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Group, PathElement, Rectangle

class LissajousExtension(inkex.EffectExtension):
    """Extension to generate Lissajous curves (harmonic oscillations)"""
    
    def add_arguments(self, pars):
        """Parameter definitions"""
        # X-axis sine wave parameters
        pars.add_argument("--freq_x", type=float, default=1.0)
        pars.add_argument("--amplitude_x", type=float, default=100.0)
        pars.add_argument("--phase_x", type=float, default=0.0)
        
        # Y-axis sine wave parameters
        pars.add_argument("--freq_y", type=float, default=1.0)
        pars.add_argument("--amplitude_y", type=float, default=100.0)
        pars.add_argument("--phase_y", type=float, default=90.0)
        
        # Drawing parameters
        pars.add_argument("--num_points", type=int, default=1000)
        pars.add_argument("--time_duration", type=float, default=10.0)
        
        # Center position
        pars.add_argument("--center_x", type=float, default=200.0)
        pars.add_argument("--center_y", type=float, default=200.0)
        
        # Style settings
        pars.add_argument("--stroke_color", default="#000000ff")
        pars.add_argument("--stroke_width", type=float, default=2.0)
        
        # Advanced options
        pars.add_argument("--smooth_curve", type=bool, default=True)
        pars.add_argument("--show_envelope", type=bool, default=False)
        
        # Preset patterns
        pars.add_argument("--preset", default="custom")
    
    def effect(self):
        """Main processing method"""
        try:
            current_layer = self.svg.get_current_layer()
            
            main_group = Group()
            main_group.label = "Lissajous Curve"
            
            self.apply_preset()
            
            if self.options.show_envelope:
                envelope = self.create_envelope()
                if envelope is not None:
                    main_group.add(envelope)
            
            curve = self.create_lissajous_curve()
            if curve is not None:
                main_group.add(curve)
            
            current_layer.add(main_group)
            
        except Exception as e:
            pass
    
    def apply_preset(self):
        """Apply preset pattern parameters"""
        presets = {
            "circle": {"freq_x": 1.0, "freq_y": 1.0, "phase_y": 90.0},
            "ellipse": {"freq_x": 1.0, "freq_y": 1.0, "phase_y": 0.0},
            "figure8": {"freq_x": 2.0, "freq_y": 1.0, "phase_y": 0.0},
            "threeleaf": {"freq_x": 3.0, "freq_y": 2.0, "phase_y": 0.0},
            "fourleaf": {"freq_x": 3.0, "freq_y": 2.0, "phase_y": 90.0},
            "star5": {"freq_x": 5.0, "freq_y": 4.0, "phase_y": 0.0},
            "complex1": {"freq_x": 7.0, "freq_y": 5.0, "phase_y": 45.0},
            "complex2": {"freq_x": 8.0, "freq_y": 7.0, "phase_y": 30.0}
        }
        
        if self.options.preset in presets:
            preset = presets[self.options.preset]
            self.options.freq_x = preset.get("freq_x", self.options.freq_x)
            self.options.freq_y = preset.get("freq_y", self.options.freq_y)
            self.options.phase_y = preset.get("phase_y", self.options.phase_y)
    
    def create_envelope(self):
        """Create envelope rectangle showing the bounds"""
        envelope = Rectangle()
        
        width = self.options.amplitude_x * 2
        height = self.options.amplitude_y * 2
        
        envelope.set('x', str(self.options.center_x - self.options.amplitude_x))
        envelope.set('y', str(self.options.center_y - self.options.amplitude_y))
        envelope.set('width', str(width))
        envelope.set('height', str(height))
        
        envelope.style = {
            'fill': 'none',
            'stroke': '#cccccc',
            'stroke-width': '1',
            'stroke-dasharray': '5,5',
            'opacity': '0.5'
        }
        
        return envelope
    
    def create_lissajous_curve(self):
        """Generate the Lissajous curve"""
        # Convert phases from degrees to radians
        phase_x_rad = math.radians(self.options.phase_x)
        phase_y_rad = math.radians(self.options.phase_y)
        
        # Calculate time step
        total_time = 2 * math.pi * self.options.time_duration
        dt = total_time / self.options.num_points
        
        # Generate points
        points = []
        for i in range(self.options.num_points + 1):
            t = i * dt
            
            # x(t) = A * sin(a*t + φ_x)
            # y(t) = B * sin(b*t + φ_y)
            x = self.options.center_x + self.options.amplitude_x * math.sin(self.options.freq_x * t + phase_x_rad)
            y = self.options.center_y + self.options.amplitude_y * math.sin(self.options.freq_y * t + phase_y_rad)
            
            points.append((x, y))
        
        # Create path
        if self.options.smooth_curve:
            path_data = self.create_smooth_path(points)
        else:
            path_data = self.create_linear_path(points)
        
        path_element = PathElement()
        path_element.set('d', path_data)
        
        stroke_color = self.options.stroke_color
        
        # Handle different color formats from Inkscape
        if isinstance(stroke_color, str):
            if stroke_color.startswith('#'):
                if len(stroke_color) == 9:
                    stroke_color = stroke_color[:7]
            else:
                try:
                    color_int = int(stroke_color)
                    # Convert RGBA integer to hex
                    r = (color_int >> 24) & 0xFF
                    g = (color_int >> 16) & 0xFF
                    b = (color_int >> 8) & 0xFF
                    # Ignore alpha channel
                    stroke_color = f"#{r:02x}{g:02x}{b:02x}"
                except (ValueError, TypeError):
                    stroke_color = "#000000"
        else:
            try:
                color_int = int(stroke_color)
                r = (color_int >> 24) & 0xFF
                g = (color_int >> 16) & 0xFF
                b = (color_int >> 8) & 0xFF
                stroke_color = f"#{r:02x}{g:02x}{b:02x}"
            except (ValueError, TypeError):
                stroke_color = "#000000"
        
        style = {
            'fill': 'none',
            'stroke': stroke_color,
            'stroke-width': str(self.options.stroke_width),
            'stroke-linejoin': 'round',
            'stroke-linecap': 'round'
        }
        
        path_element.style = style
        
        return path_element
    
    def create_linear_path(self, points):
        """Create linear path from points"""
        if not points:
            return ""
        
        path_data = [f"M {points[0][0]},{points[0][1]}"]
        
        for x, y in points[1:]:
            path_data.append(f"L {x},{y}")
        
        return " ".join(path_data)
    
    def create_smooth_path(self, points):
        """Create smooth Bezier path from points"""
        if len(points) < 4:
            return self.create_linear_path(points)
        
        path_data = [f"M {points[0][0]},{points[0][1]}"]
        
        # Create smooth curve using cubic Bezier
        for i in range(1, len(points) - 2):
            p0 = points[i - 1] if i > 0 else points[i]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2] if i + 2 < len(points) else points[i + 1]
            
            tension = 0.5
            
            cp1_x = p1[0] + (p2[0] - p0[0]) * tension / 6
            cp1_y = p1[1] + (p2[1] - p0[1]) * tension / 6
            
            cp2_x = p2[0] - (p3[0] - p1[0]) * tension / 6
            cp2_y = p2[1] - (p3[1] - p1[1]) * tension / 6
            
            path_data.append(f"C {cp1_x},{cp1_y} {cp2_x},{cp2_y} {p2[0]},{p2[1]}")
        
        return " ".join(path_data)
    
    def calculate_optimal_duration(self):
        """Calculate optimal time duration for complete pattern"""
        # Find LCM of frequencies to get complete cycle
        from math import gcd
        
        scale = 100
        freq_x_int = int(self.options.freq_x * scale)
        freq_y_int = int(self.options.freq_y * scale)
        
        lcm = abs(freq_x_int * freq_y_int) // gcd(freq_x_int, freq_y_int)
        optimal_duration = lcm / scale
        
        return min(optimal_duration, 50.0)  # Cap at 50 cycles

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
        
        extension = LissajousExtension()
        extension.run()
        
    except Exception as e:
        pass