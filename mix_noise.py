from pydub import AudioSegment
import os
import random
import easygui
from pydub.effects import normalize
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_noise_level(voice, noise, target_dBFS_difference=-15):
    """
    Calculates the noise level adjustment needed to achieve a target dBFS difference from the voice to the noise.
    """
    voice_dBFS = voice.dBFS
    noise_dBFS = noise.dBFS
    return noise_dBFS - voice_dBFS + target_dBFS_difference

def extend_to_one_minute(audio_segment):
    """
    Extends or trims an audio segment to exactly one minute.
    """
    one_minute = 60 * 1000  # 60 seconds in milliseconds
    if len(audio_segment) < one_minute:
        return (audio_segment * (one_minute // len(audio_segment) + 1))[:one_minute]
    else:
        return audio_segment[:one_minute]

def mix_audio_files(voice_file_path, noise_file_path, output_dir, target_dBFS_difference):
    """
    Mixes a voice file with a background noise file and saves the outputs in a specified directory.
    """
    try:
        voice = normalize(AudioSegment.from_file(voice_file_path))
        noise = normalize(AudioSegment.from_file(noise_file_path))

        # Calculate dynamic noise level
        noise_level = calculate_noise_level(voice, noise, target_dBFS_difference)

        # Adjust noise volume
        noise = noise + noise_level

        # Ensure both segments are one minute long
        voice = extend_to_one_minute(voice)
        noise = extend_to_one_minute(noise)

        # Mix the voice with the background noise
        mixed = voice.overlay(noise)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Export the mixed and original voice audio files
        mixed.export(os.path.join(output_dir, "mixed_output.wav"), format='wav')
        voice.export(os.path.join(output_dir, "voice_output.wav"), format='wav')
        
        # Save metadata
        metadata = {
            "voice_file": voice_file_path,
            "noise_file": noise_file_path,
            "target_dBFS_difference": target_dBFS_difference,
            "noise_level": noise_level
        }
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Files have been saved in {output_dir}")
    except Exception as e:
        logging.error(f"Error mixing audio files: {e}")

def find_random_files(directory, file_extension=".wav", count=1):
    """
    Returns a list of random file paths from a directory, including its subdirectories, up to the specified count.
    """
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(file_extension):
                matches.append(os.path.join(root, filename))
    return random.sample(matches, min(count, len(matches))) if matches else []

def main_menu():
    logging.info("Audio Mixing Tool Started")
    voice_dir = easygui.diropenbox(title="Select the directory containing voice WAV files")
    noise_dir = easygui.diropenbox(title="Select the directory containing noise WAV files")
    output_root_dir = easygui.diropenbox(title="Select the root output directory")
    number_of_mixes = easygui.integerbox("Enter the number of mixed files you want to create:", lowerbound=1, upperbound=1000, default=1)
    
    if not voice_dir or not noise_dir or not output_root_dir or not number_of_mixes:
        logging.error("Missing input. Please provide all necessary inputs.")
        return

    target_dBFS_differences = [-10, -15, -20]  # Different scenarios
    
    # Randomly pick specified number of voice and noise files
    voice_files = find_random_files(voice_dir, count=number_of_mixes)
    noise_files = find_random_files(noise_dir, count=number_of_mixes)

    if not voice_files or not noise_files:
        logging.error("Could not find enough WAV files in one or both directories.")
        return

    for counter, (voice_file_path, noise_file_path) in enumerate(zip(voice_files, noise_files), start=1):
        for target_dBFS_difference in target_dBFS_differences:
            try:
                output_dir = os.path.join(output_root_dir, f"output_sample_{counter}_dBFS_{target_dBFS_difference}")
                mix_audio_files(voice_file_path, noise_file_path, output_dir, target_dBFS_difference)
            except Exception as e:
                logging.error(f"Failed to process files {voice_file_path} and {noise_file_path} with target_dBFS_difference {target_dBFS_difference}: {e}")
                continue

if __name__ == "__main__":
    main_menu()
