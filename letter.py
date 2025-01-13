import os
import logging
from typing import List, Optional
import numpy as np
import svgwrite
import drawing
from rnn import rnn

class Hand(object):

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)

        line_height = 60
        view_width = 1000
        view_height = line_height*(len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3*line_height / 4)])
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height

        dwg.save()


class ModularHandwriter:
    def __init__(self, base_hand):
        """Initialize with an existing Hand instance"""
        self.hand = Hand()
        
    def write_letter_section(
        self,
        text: str,
        output_filename: str,
        style: int = 7,
        bias: float = 0.75,
        stroke_color: str = 'black',
        stroke_width: int = 2
    ):
        """Write a single section of text to an SVG file"""
        lines = [text]  # Single line of text
        biases = [bias]
        styles = [style]
        stroke_colors = [stroke_color]
        stroke_widths = [stroke_width]
        
        self.hand.write(
            filename=output_filename,
            lines=lines,
            biases=biases,
            styles=styles,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths
        )

    def write_complete_letter(
        self,
        salutation: str,
        addressing: str,
        body: str,
        signature: Optional[str] = None,
        output_dir: str = 'output',
        base_filename: str = 'letter',
        style_map: Optional[dict] = None,
        bias_map: Optional[dict] = None
    ):
        """Write each section of the letter to separate files and optionally combine them"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Default style and bias maps if none provided
        default_style_map = {
            'salutation': 7,  # More formal style
            'addressing': 7,  # Consistent with salutation
            'body': 7,       # Main content style
            'signature': 8    # Slightly more personal style
        }
        
        default_bias_map = {
            'salutation': 0.75,  # More consistent writing
            'addressing': 0.65,  # Slightly more variation
            'body': 0.55,        # More natural variation
            'signature': 0.85    # Very consistent signature
        }
        
        style_map = style_map or default_style_map
        bias_map = bias_map or default_bias_map
        
        # Generate individual sections
        sections = {
            'salutation': salutation,
            'addressing': addressing,
            'body': body
        }
        if signature:
            sections['signature'] = signature
            
        # Write each section
        output_files = {}
        for section_name, content in sections.items():
            output_filename = os.path.join(output_dir, f"{base_filename}_{section_name}.svg")
            self.write_letter_section(
                text=content,
                output_filename=output_filename,
                style=style_map[section_name],
                bias=bias_map[section_name]
            )
            output_files[section_name] = output_filename
            
        return output_files

# Usage example
if __name__ == "__main__":
    
    # Initialize the base hand and our modular writer
    base_hand = Hand()
    writer = ModularHandwriter(base_hand)
    
    # Define letter content
    letter_content = {
        'salutation': "Dear John,",
        'addressing': "Thank you for reaching out to me regarding the project proposal.",
        'body': "Here is the detailed information you requested about our implementation.",
        'signature': "Best regards, Jane Smith"
    }
    
    # Custom style and bias maps (optional)
    custom_style_map = {
        'salutation': 7,
        'addressing': 7,
        'body': 7,
        'signature': 8
    }
    
    custom_bias_map = {
        'salutation': 0.8,
        'addressing': 0.7,
        'body': 0.6,
        'signature': 0.9
    }
    
    # Generate the letter sections
    output_files = writer.write_complete_letter(
        salutation=letter_content['salutation'],
        addressing=letter_content['addressing'],
        body=letter_content['body'],
        signature=letter_content['signature'],
        output_dir='output/letter_sections',
        base_filename='business_letter',
        style_map=custom_style_map,
        bias_map=custom_bias_map
    )