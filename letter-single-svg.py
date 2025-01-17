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

    def write_combined(self, sections, filename, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        """Write multiple text sections to a single SVG file with proper spacing"""
        valid_char_set = set(drawing.alphabet)
        all_lines = []
        all_biases = []
        all_styles = []
        all_colors = []
        all_widths = []
        
        # Process each section into individual lines
        for i, section in enumerate(sections):
            # Split section into lines and strip whitespace
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            
            for line in lines:
                if len(line) > 75:
                    raise ValueError(f"Each line must be at most 75 characters. Line contains {len(line)}: {line}")
                for char in line:
                    if char not in valid_char_set:
                        raise ValueError(f"Invalid character {char} detected. Valid character set is {valid_char_set}")
                
                all_lines.append(line)
                all_biases.append(biases[i] if biases else 0.5)
                all_styles.append(styles[i] if styles else None)
                all_colors.append(stroke_colors[i] if stroke_colors else 'black')
                all_widths.append(stroke_widths[i] if stroke_widths else 2)

        # Generate strokes for all lines
        strokes = self._sample(all_lines, biases=all_biases, styles=all_styles)
        
        # Draw combined SVG
        self._draw_combined(strokes, all_lines, filename, stroke_colors=all_colors, stroke_widths=all_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(f'styles/style-{style}-strokes.npy')
                c_p = np.load(f'styles/style-{style}-chars.npy').tostring().decode('utf-8')

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

    def _draw_combined(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        """Draw all sections in a single SVG with appropriate spacing"""
        stroke_colors = stroke_colors or ['black'] * len(lines)
        stroke_widths = stroke_widths or [2] * len(lines)

        # Adjust spacing between sections
        line_height = 60
        view_width = 1000
        view_height = line_height * (len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3 * line_height / 4)])
        
        for i, (offsets, line, color, width) in enumerate(zip(strokes, lines, stroke_colors, stroke_widths)):
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
        self.hand = Hand()
    
    def write_complete_letter(
        self,
        salutation: str,
        addressing: str,
        body: str,
        signature: Optional[str] = None,
        output_filename: str = 'complete_letter.svg',
        style_map: Optional[dict] = None,
        bias_map: Optional[dict] = None
    ):
        """Generate a complete letter in a single SVG file"""
        default_style_map = {
            'salutation': 7,
            'addressing': 7,
            'body': 7,
            'signature': 8
        }
        
        default_bias_map = {
            'salutation': 0.75,
            'addressing': 0.65,
            'body': 0.55,
            'signature': 0.85
        }
        
        style_map = style_map or default_style_map
        bias_map = bias_map or default_bias_map
        
        # Prepare sections
        sections = [salutation, addressing, body]
        if signature:
            sections.append(signature)
            
        # Prepare styles and biases
        styles = [style_map[section] for section in ['salutation', 'addressing', 'body'] + (['signature'] if signature else [])]
        biases = [bias_map[section] for section in ['salutation', 'addressing', 'body'] + (['signature'] if signature else [])]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Generate single SVG with all sections
        self.hand.write_combined(
            sections=sections,
            filename=output_filename,
            styles=styles,
            biases=biases
        )
        
        return output_filename

# Usage example
if __name__ == "__main__":
    base_hand = Hand()
    writer = ModularHandwriter(base_hand)
    
    # Define letter content with multiple lines, each under 75 characters
    letter_content = {
        'salutation': "Dear Dr. Smith,\n123 Innovation Drive,\nSilicon Valley, CA 94025",
        
        'addressing': "I hope this letter finds you well.\nI am writing regarding our recent discussion about the AI project.",
        
        'body': "I have reviewed the preliminary results you shared during our last meeting.\n\n"
                "The neural network architecture you proposed shows remarkable promise,\n"
                "particularly in handling the edge cases we discussed. Your innovative\n"
                "approach to feature extraction could be groundbreaking.\n\n"
                "I would like to schedule a follow-up meeting next week to discuss\n"
                "potential improvements and next steps. Would Tuesday or Wednesday\n"
                "afternoon work for your schedule?\n\n"
                "Additionally, I've attached the latest performance metrics and\n"
                "comparative analysis you requested. Please review them at your convenience.",
        
        'signature': "Best regards,\nDr. Jane Anderson\nDirector of AI Research\nTech Innovations Lab"
    }
    
    # Generate the complete letter in a single SVG
    output_file = writer.write_complete_letter(
        salutation=letter_content['salutation'],
        addressing=letter_content['addressing'],
        body=letter_content['body'],
        signature=letter_content['signature'],
        output_filename='output/complete_letter.svg'
    )