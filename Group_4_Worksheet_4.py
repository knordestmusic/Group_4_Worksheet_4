import numpy as np
import pretty_midi
from scipy.io.wavfile import write
from scipy.signal import freqz
import soundfile as sf
from SchroederReverb import SchroederReverb
from matplotlib import pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile


def Section_1_1():
    """FM SYNTHESIS"""

    midi_path = "MIDI.mid"  # arranged in a daw and exported to be used with pretty midi
    fs = 48000
    mod_index = 2
    harmonic_ratio = 2.5  # modulation index in terms of harmonic ratio of modulator frequency to carrier frequency to get musically coherent notes
    decay_rate = 1.2
    min_cutoff = 250  # minimum cutoff frequency for LPF
    max_cutoff_mul = (
        1.3  # maximum cutoff as a multiple of the carrier frequency for variable LPF
    )
    output_wav = "fm_synthesis_output.wav"

    # pretty MIDI is pretty cool!
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_duration = midi_data.get_end_time()
    audio = np.zeros(int(fs * total_duration))
    pre_filter = np.zeros(int(fs * total_duration))

    # FM per note
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            duration = end_time - start_time
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            num_samples = end_idx - start_idx
            t = np.linspace(0, duration, num_samples, endpoint=False)

            envelope = np.exp(-decay_rate * t)

            # alotting frequencies to the notes, one note at time
            f0 = pretty_midi.note_number_to_hz(note.pitch)
            fm = harmonic_ratio * f0
            beta = mod_index

            # FM
            modulator = np.sin(2 * np.pi * fm * t)
            carrier = np.sin(2 * np.pi * f0 * t + beta * modulator)
            signal = carrier * envelope * (note.velocity / 127.0)

            pre_filter[start_idx:end_idx] += signal

            # low-pass filter
            y = np.zeros_like(signal)
            y_prev = 0.0
            for i in range(len(signal)):
                fc = min_cutoff + envelope[i] * (f0 * max_cutoff_mul - min_cutoff)
                alpha = (2 * np.pi * fc / fs) / (2 * np.pi * fc / fs + 1)
                y[i] = y_prev + alpha * (signal[i] - y_prev)
                y_prev = y[i]

            audio[start_idx:end_idx] += y

    audio /= np.max(np.abs(audio))
    audio *= 0.9

    write(output_wav, fs, (audio * 32767).astype(np.int16))

    fm, fs = sf.read("fm_synthesis_output.wav")
    
    def compute_spectrum(x):
        X    = np.fft.rfft(x)
        freq = np.fft.rfftfreq(len(x), 1/fs)
        mag  = np.abs(X)
        return freq, mag

    f_mod, mag_mod = compute_spectrum(modulator)
    f_car, mag_car = compute_spectrum(carrier)
    f_sig, mag_sig = compute_spectrum(signal)
    f_out, mag_out = compute_spectrum(y)

    # STFT for Synthesized Signal
    f_fm, t_fm, Zxx_fm = stft(
        audio, fs=fs, window="hann", nperseg=4096, noverlap=2048, boundary=None
    )
    magnitude_fm = 20 * np.log10(np.abs(Zxx_fm) + 1e-6)

    # STFT Synthesised signal before filter aplication
    f_filter, t_filter, Zxx_filter = stft(
        pre_filter,
        fs=fs,
        window="hann",
        nperseg=4096,
        noverlap=2048,
        boundary=None,
    )
    magnitude_filter = 20 * np.log10(np.abs(Zxx_filter) + 1e-6)

    # To limit STFT between 20 Hz to 10000Hz
    freq_mask_fm = (f_fm >= 20) & (f_fm <= 10000)
    f_fm_trunc = f_fm[freq_mask_fm]
    mag_fm_trunc = magnitude_fm[freq_mask_fm, :]
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t_fm, f_fm_trunc, mag_fm_trunc, shading="gouraud")
    plt.ylim(20, 10000)
    plt.title("FM synthesized Audio Spectrum")
    plt.colorbar(label="Magnitude (dB)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("STFT Spectrogram")
    plt.tight_layout()
    plt.show()


    # To limit STFT between 20 Hz to 10000Hz
    freq_mask_filt = (f_filter >= 20) & (f_filter <= 10000)
    f_filt_trunc = f_filter[freq_mask_filt]
    mag_filt_trunc = magnitude_filter[freq_mask_filt, :]
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t_filter, f_filt_trunc, mag_filt_trunc, shading="gouraud")
    plt.ylim(20, 10000)
    plt.title("Pre-Filter FM Synthesized Audio Spectrum")
    plt.colorbar(label="Magnitude (dB)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("STFT Spectrogram")
    plt.tight_layout()
    plt.show()

    # Plot: Modulator Spectrum
    plt.figure()
    plt.plot(f_mod, mag_mod)
    plt.title('Modulator Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0, 10000)
    plt.ylabel('Magnitude')

    # Plot: Carrier Spectrum
    plt.figure()
    plt.plot(f_car, mag_car)
    plt.title('Carrier Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 10000)

    # Plot: Signal Spectrum (Post-Envelope)
    plt.figure()
    plt.plot(f_sig, mag_sig)
    plt.title('Signal Spectrum (Post-Envelope)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 10000)

    # Plot: Output Spectrum (Post-Filter)
    plt.figure()
    plt.plot(f_out, mag_out)
    plt.title('Output Spectrum (Post-Filter)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 10000)

    #Filter Response
    
    fc_min = min_cutoff
    fc_max = f0 * max_cutoff_mul
    alpha_min = (2 * np.pi * fc_min / fs) / (2 * np.pi * fc_min / fs + 1)
    alpha_max = (2 * np.pi * fc_max / fs) / (2 * np.pi * fc_max / fs + 1)
    b_min, a_min = [alpha_min], [1, -(1 - alpha_min)]
    b_max, a_max = [alpha_max], [1, -(1 - alpha_max)]

    w, h_min = freqz(b_min, a_min, worN=1024, fs=fs)
    _, h_max = freqz(b_max, a_max, worN=1024, fs=fs)

    # Filter Frequency Response
    plt.figure()
    plt.plot(w, np.abs(h_min))
    plt.plot(w, np.abs(h_max))
    plt.title("Dynamic One-Pole LPF Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.legend(["Cutoff = min_cutoff", "Cutoff = max_cutoff"])
    plt.tight_layout()
    plt.show()


    reverb = SchroederReverb(
        fs=fs,
        comb_delays=[1000, 3000, 1500, 2000],
        comb_gains=[0.8, 0.8, 0.7, 0.75],
        ap_delays=[500, 600, 850, 650],
        ap_gain=0.6,
    )

    dry_level = 0.5
    wet_level = 0.8

    def apply_reverb(dry):
        wet = reverb.process(dry)
        out = dry_level * dry + wet_level * wet
        # avoid clipping
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak
        return out

    fm_rev = apply_reverb(fm)
    sf.write("fm_with_reverb.wav", fm_rev, fs)


# ----------------------------########----------------------------########----------------------------#


def Section_1_2():
    """WAVEGUIDE SYNTHESIS"""

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

    fm, fs = sf.read("waveGuide_Synthesis.wav")
    
    sample_rate, audio_data = wavfile.read('waveGuide_Synthesis.wav')
    
    # STFT Synthesised signal
    f, t, Zxx = stft(
        audio_data, fs=sample_rate, window="hann", nperseg=4096, noverlap=2048, boundary=None
    )

    mag = 20 * np.log10(np.abs(Zxx) + 1e-6)
    
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, mag, shading="gouraud")
    plt.title("STFT of waveguide synthesis output signal")
    plt.colorbar(label="Magnitude (dB)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("STFT Spectrogram")
    plt.tight_layout()
    plt.ylim(0, sample_rate/2)
    plt.show()
    
    
    reverb = SchroederReverb(
        fs=fs,
        comb_delays=[1000, 3000, 1500, 2000],
        comb_gains=[0.8, 0.8, 0.7, 0.75],
        ap_delays=[500, 600, 850, 650],
        ap_gain=0.6,
    )

    dry_level = 0.5
    wet_level = 0.8

    def apply_reverb(dry):
        wet = reverb.process(dry)
        out = dry_level * dry + wet_level * wet
        # avoid clipping
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak
        return out

    fm_rev = apply_reverb(fm)
    sf.write("waveguide_with_reverb.wav", fm_rev, fs)


Section_1_1()  # FOR CALLING THE RESULTS OF FM SYNTHESIS SECTION
Section_1_2()  # FOR CALLING THE RESULTS OF WAVEGUIDE SYNTHESIS SECTION
