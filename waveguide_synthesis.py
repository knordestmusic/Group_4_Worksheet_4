import numpy as np
import pretty_midi
import soundfile as sf
import matplotlib.pyplot as plt
from waveguide_module import waveGuide 


def synthesize_midi(
    midi_file_path,
    output_file_path="output.wav",
    pickup_ratio=0.8,
    pluck_ratio=0.2,
    base_loss_factor=0.995,
    loss_variation=0.01,
    filter_size=6,
):

    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    fs = 48000

    duration = midi_data.get_end_time()
    total_samples = int(duration * fs) + fs  # added extra second for safety
    output = np.zeros(total_samples)

    total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
    notes_processed = 0

    # Process each instrument
    for i, instrument in enumerate(midi_data.instruments):

        for note in instrument.notes:
            # to calculate frequency from MIDI note number
            frequency = pretty_midi.note_number_to_hz(note.pitch)

            # to calculate waveguide length based on frequency
            L = int(fs / (2 * frequency))
            L = max(10, L)

            # Create waveguide on per note basis
            pickup_position = int(L * pickup_ratio)
            loss_factor = base_loss_factor - (note.pitch / 127) * loss_variation
            wg = waveGuide(L, pickup_position, loss_factor, filter_size)

            pluck_position = int(L * pluck_ratio)
            wg.pluck(pluck_position)

            # note duration in samples
            start_sample = int(note.start * fs)
            end_sample = int(note.end * fs)
            note_duration_samples = end_sample - start_sample

            amplitude = note.velocity / 127.0

            # Generate audio on per note basis
            note_buffer = np.zeros(note_duration_samples)
            for j in range(note_duration_samples):
                note_buffer[j] = wg.wavePropagation() * amplitude

            # Add to the output buffer
            if start_sample + note_duration_samples <= len(output):
                output[
                    start_sample : start_sample + note_duration_samples
                ] += note_buffer

    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.9

    sf.write(output_file_path, output, fs)
    print(f"Audio saved to {output_file_path}")

    return output


midi_file = "MIDI.mid"

synthesize_midi(
    midi_file,
    "waveGuide_Synthesis.wav",
    pickup_ratio=0.9,
    pluck_ratio=0.5,
    base_loss_factor=0.998,
    loss_variation=0.005,
)
